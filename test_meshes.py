"""Utilities for domain and mesh creation"""

import sys
import time
from itertools import combinations
import matplotlib.pyplot as plt
import meshio

import firedrake as fd
import pickle

import gmsh
import numpy as np
import scipy.stats.qmc as qmc
from colorama import Fore
from firedrake import COMM_WORLD

import config
from utils.parallel import global_print

sys.path.append("..")

nproc = COMM_WORLD.size
rank = COMM_WORLD.rank


def distance(x1: float, y1: float, x2: float, y2: float):
    """Computes the Euclidean distance between points (x1, y1) and (x2, y2)"""
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def visualize_mesh_transfinite(meshfile):
    mesh = meshio.read(meshfile)

    # print(mesh)

    # Plot the mesh
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot vertices
    ax.scatter(mesh.points[:, 0], mesh.points[:, 1], mesh.points[:, 2], color="r", s=1)

    # Plot cells
    for cells in mesh.cells:
        if cells.type == "line":
            for line in cells.data:
                vertices = [mesh.points[i] for i in line]
                ax.plot(
                    [vertices[0][0], vertices[1][0]],
                    [vertices[0][1], vertices[1][1]],
                    [vertices[0][2], vertices[1][2]],
                    color="b",
                )
        elif cells.type == "quad":
            for quad in cells.data:
                vertices = [mesh.points[i] for i in quad]
                vertices.append(mesh.points[quad[0]])  # Close the quad loop
                ax.plot(
                    [vertex[0] for vertex in vertices],
                    [vertex[1] for vertex in vertices],
                    [vertex[2] for vertex in vertices],
                    color="g",
                )

    # Add additional visualization for cell sets, point/cell data if needed

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()


def visualize_mesh(meshfile):
    mesh = meshio.read(meshfile)

    # Plot the mesh
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot vertices
    ax.scatter(mesh.points[:, 0], mesh.points[:, 1], mesh.points[:, 2], color="r", s=1)

    # Plot cells
    for cells in mesh.cells:
        if cells.type == "line":
            for line in cells.data:
                vertices = [mesh.points[i] for i in line]
                ax.plot(
                    [vertices[0][0], vertices[1][0]],
                    [vertices[0][1], vertices[1][1]],
                    [vertices[0][2], vertices[1][2]],
                    color="b",
                )
        elif cells.type == "triangle":
            for quad in cells.data:
                vertices = [mesh.points[i] for i in quad]
                vertices.append(mesh.points[quad[0]])  # Close the quad loop
                ax.plot(
                    [vertex[0] for vertex in vertices],
                    [vertex[1] for vertex in vertices],
                    [vertex[2] for vertex in vertices],
                    color="g",
                )

    # Add additional visualization for cell sets, point/cell data if needed

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()


def create_transfinite_domain(
    buffer_length: float,
    lx: float,
    ly: float,
    DIVISIONS,
    meshfile: str,
):
    r"""
    Create transfinite domain for Fourier Neural Operator.

    Args:
        buffer_length (float): length of the inlet and outlet branches, meaning the lateral deviation from
                               the coordinates passed to inlet_positions and outlet_positions
        lx (float): x-length of the design domain
        ly (float): y-length of the design domain
        DIVISIONS (int): number of divisions to use in the mesh
        meshfile (str): .msh file to be imported as a mesh, this is created before the start of the optimization
                        by the method create_domain in utils/create_domain.py

    Return:
        inlet (dict): a dictionary of the inlets with key given by the ID of the inlet to use in Firedrake
                      and pair a tuple given by (sign of the outward pressure direction depending on the position of
                      the inlet, component of the test function which is not cancelled out in the weak formulation)
        outlet (list): a list with the IDs of the outlets to use in Firedrake
        DESIGN (int, optional): ID to utilize for the design domain, namely the inner rectangle [0, lx] \times [0, ly].
                                Defaults to 3
        NON_DESIGN (int, optional): ID to utilize for the non design domain defaults to 1
    """
    DESIGN: int = 3
    NON_DESIGN: int = 1
    if rank == 0:
        mesh_time = time.time()

        # Start the Gmsh API
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)

        # Create a new Gmsh model
        gmsh.model.add("domain_model")

        # Create the main rectangle (domain)
        domain = gmsh.model.occ.addRectangle(0, 0, 0, lx, ly)

        # Create inlets and outlets
        inlet = gmsh.model.occ.addRectangle(-buffer_length, 0, 0, buffer_length, ly)
        outlet = gmsh.model.occ.addRectangle(lx, 0, 0, buffer_length, ly)

        # Fragment the domain and inlets/outlets
        gmsh.model.occ.fragment([(2, domain)], [(2, inlet), (2, outlet)])

        # Synchronize model with OCC to finalize the changes
        gmsh.model.occ.synchronize()

        lines = gmsh.model.occ.getEntities(dim=1)

        inlet = {}
        outlet = []
        walls = []
        non_walls = []

        for line_dim, line_tag in lines:
            # get center of mass and length of the edge
            com = gmsh.model.occ.getCenterOfMass(line_dim, line_tag)

            if np.isclose(com[0], -buffer_length) and np.isclose(com[1], ly / 2.0):
                inlet[line_tag] = (1, 0)

            if np.isclose(com[0], lx + buffer_length) and np.isclose(com[1], ly / 2.0):
                outlet.append(line_tag)

            if (np.isclose(com[0], 0) or np.isclose(com[0], lx)) and np.isclose(
                com[1], ly / 2.0
            ):
                non_walls.append(line_tag)

        # Get all the surfaces after fragmentation
        surfaces = gmsh.model.occ.getEntities(dim=2)

        # Set the transfinite line mesh on each edge of each surface
        divisions = DIVISIONS
        divisions_buffer = int(divisions * buffer_length / (2 * lx))

        fluid = [surface_tag for _, surface_tag in surfaces]
        non_design_domain = [
            surface_tag for _, surface_tag in surfaces if surface_tag != domain
        ]
        domain = [surface_tag for _, surface_tag in surfaces if surface_tag == domain]
        gmsh.model.addPhysicalGroup(
            2, non_design_domain, NON_DESIGN, "non_design_domain"
        )
        gmsh.model.addPhysicalGroup(2, domain, DESIGN, "domain")
        gmsh.model.addPhysicalGroup(2, fluid, -1, "fluid")

        for surface in surfaces:
            boundaries = gmsh.model.getBoundary([surface], oriented=False)
            for curve_dim_tag in boundaries:
                curve_tag = curve_dim_tag[1]  # get the tag
                COM = gmsh.model.occ.getCenterOfMass(1, curve_tag)
                if np.isclose(COM[1], ly / 2.0) or np.isclose(COM[0], lx / 2.0):
                    gmsh.model.mesh.setTransfiniteCurve(curve_tag, divisions)
                else:
                    gmsh.model.mesh.setTransfiniteCurve(curve_tag, divisions_buffer)

            # Set a transfinite surface mesh on each surface
            gmsh.model.mesh.setTransfiniteSurface(surface[1])

        # get walls as all lines which are not inlets nor outlets nor invalid walls
        walls = [
            line_tag
            for _, line_tag in lines
            if line_tag not in inlet
            and line_tag not in outlet
            and line_tag not in non_walls
        ]

        # create physical group for each inlet
        for key in inlet.keys():
            gmsh.model.addPhysicalGroup(1, [key], key)

        # create physical group for each outlet
        for key in outlet:
            gmsh.model.addPhysicalGroup(1, [key], key)

        # create unique "outlet", "walls" and "non_walls" physical groups, the latter mostly for checks
        gmsh.model.addPhysicalGroup(1, walls, 3, "walls")
        gmsh.model.addPhysicalGroup(1, non_walls, 100, "non_walls")

        # Synchronize the model
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.RecombineAll", 1)

        # Generate the 2D mesh
        gmsh.model.mesh.generate(2)

        # Save the mesh to a file
        global_print(f"Saving mesh to {config.meshdir + meshfile}")
        gmsh.write(config.meshdir + meshfile)

        # Finalize Gmsh API
        gmsh.finalize()

        global_print(
            f"{Fore.GREEN}Transfinite Domain generation .............. {time.time()-mesh_time : 2.2f} s{Fore.RESET}"
        )
    else:
        inlet = None
        outlet = None

    inlet = COMM_WORLD.bcast(inlet, root=0)
    outlet = COMM_WORLD.bcast(outlet, root=0)

    return inlet, outlet, DESIGN, NON_DESIGN


def print_mesh_content(meshfile):
    # Load mesh from file
    with open(meshfile, "r") as f:
        lines = f.readlines()

    # Print the content of the mesh file
    for line in lines:
        print(line.strip())


def create_inlets_outlets(
    inlet_positions: list,
    outlet_positions: list,
    buffer_length: float,
    lx: float,
    ly: float,
):
    r"""Creates inlets and outlets

    Args:
        inlet_positions (list): list with inlets coordinates
        outlet_positions (list): list with outlets coordinates
        buffer_length (float): branch length popping out of the design domain [0, lx] \times [0, ly]
        lx (float): x-length of the design domain
        ly (float): y-length of the design domain

    Returns:
        inlets (list): list of inlets rectangles
        outlets (list): list of outlets rectangles
    """
    inlets = []
    outlets = []

    # Create inlet(s)
    for inlet_position in inlet_positions:
        x1, y1, x2, y2 = inlet_position
        assert (
            0 <= x1 <= lx and 0 <= x2 <= lx and 0 <= y1 <= ly and 0 <= y2 <= ly
        ), f"Passed inlet {inlet_position} is not valid, given the dimensions of the chip"
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0.0:
            assert x1 == 0 or x1 == lx
            if x1 == 0.0:
                inlet = gmsh.model.occ.addRectangle(
                    -buffer_length, y1, 0, buffer_length, dy
                )
            else:
                inlet = gmsh.model.occ.addRectangle(lx, y1, 0, buffer_length, dy)
        elif dy == 0.0:
            assert y1 == 0 or y1 == ly
            if y1 == 0.0:
                inlet = gmsh.model.occ.addRectangle(
                    x1, -buffer_length, 0, dx, buffer_length
                )
            else:
                inlet = gmsh.model.occ.addRectangle(x1, ly, 0, dx, buffer_length)
        inlets.append(inlet)

    # Create outlet(s)
    for outlet_position in outlet_positions:
        x1, y1, x2, y2 = outlet_position
        assert (
            0 <= x1 <= lx and 0 <= x2 <= lx and 0 <= y1 <= ly and 0 <= y2 <= ly
        ), f"Passed outlet {outlet_position} is not valid, given the dimensions of the chip"
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0.0:
            assert x1 == 0 or x1 == lx
            if x1 == 0.0:
                outlet = gmsh.model.occ.addRectangle(
                    -buffer_length, y1, 0, buffer_length, dy
                )
            else:
                outlet = gmsh.model.occ.addRectangle(lx, y1, 0, buffer_length, dy)
        elif dy == 0.0:
            assert y1 == 0 or y1 == ly
            if y1 == 0.0:
                outlet = gmsh.model.occ.addRectangle(
                    x1, -buffer_length, 0, dx, buffer_length
                )
            else:
                outlet = gmsh.model.occ.addRectangle(x1, ly, 0, dx, buffer_length)
        outlets.append(outlet)

    return inlets, outlets


def boundary_splitting(
    inlet_positions: list,
    outlet_positions: list,
    buffer_length: float,
    lx: float,
    ly: float,
):
    r"""Boundary splitting between inlet, outlet, walls and non_walls (interfaces)

    Args:
        inlet_positions (list): list with inlets coordinates
        outlet_positions (list): list with outlets coordinates
        buffer_length (float): branch length popping out of the design domain [0, lx] \times [0, ly]
        lx (float): x-length of the design domain
        ly (float): y-length of the design domain

    Returns:
        inlet (dict): dictionary with keys the IDs of the inlet and values a tuple with:
                            1. sign of the normal to the boundary (1 or -1)
                            2. component of the test function which is not cancelled out by
                               the previous sign in the computation (0 or 1)
        outlet (list): list of outlets IDs
        non_walls (list): list of interfaces between branches and design domain to be cancelled out from the walls for
                          correct BCs imposition
    """
    lines = gmsh.model.occ.getEntities(dim=1)

    inlet = {}
    outlet = []
    non_walls = []
    # cycle over the lines
    for line_dim, line_tag in lines:
        update_inlet_dict(
            inlet, line_dim, line_tag, inlet_positions, buffer_length, lx, ly
        )
        update_outlet_dict(
            outlet, line_dim, line_tag, outlet_positions, buffer_length, lx, ly
        )

    get_non_walls(non_walls, lines, inlet_positions, outlet_positions)

    return inlet, outlet, non_walls


def update_inlet_dict(
    inlet: dict,
    line_dim: int,
    line_tag: int,
    inlet_positions: list,
    buffer_length: float,
    lx: float,
    ly: float,
):
    """Update inlet dictionary with direction of outward normal for BCs imposition and component of test function
       to be used for pressure drop

    Args:
        inlet (dict): inlet dictionary to be updated
        line_dim (int): always 1
        line_tag (int): tag of the current line under examination
        inlet_positions (list): list with all the inlet positions
        buffer_length (float): buffer length
        lx (float): x-length of the design domain
        ly (float): y-length of the design domain
    """
    # cycle over the inlets (compute sign and component of stress which needs to be set in BCs)
    # get center of mass and length of the edge
    com = gmsh.model.occ.getCenterOfMass(line_dim, line_tag)
    mass = gmsh.model.occ.getMass(line_dim, line_tag)

    for inlet_position in inlet_positions:
        x1, y1, x2, y2 = inlet_position
        if x1 == 0 and x2 == 0:  # this inlet is on the left edge
            component = 0
            sign = 1
            com_inlet = [-buffer_length, (y1 + y2) / 2.0]
        elif y1 == 0 and y2 == 0:  # this inlet is on the bottom edge
            component = 1
            sign = 1
            com_inlet = [(x1 + x2) / 2.0, -buffer_length]
        elif y1 == ly and y2 == ly:  # this inlet is on the top edge
            component = 1
            sign = -1
            com_inlet = [(x1 + x2) / 2.0, ly + buffer_length]
        elif x1 == lx and x2 == lx:  # this inlet is on the right edge
            component = 0
            sign = -1
            com_inlet = [lx + buffer_length, (y1 + y2) / 2.0]

        mass_inlet = distance(x1, y1, x2, y2)

        # add inlet to the inlets IDs
        if (
            np.isclose(com[0], com_inlet[0])
            and np.isclose(com[1], com_inlet[1])
            and np.isclose(mass, mass_inlet)
            and line_tag not in inlet
        ):
            inlet[line_tag] = (sign, component)


def update_outlet_dict(
    outlet: list,
    line_dim: int,
    line_tag: int,
    outlet_positions: list,
    buffer_length: float,
    lx: float,
    ly: float,
):
    """Update outlet list

    Args:
        outlet (list): list to be updated
        line_dim (int): always 1
        line_tag (int): tag of the current line under examination
        outlet_positions (list): list with all the outlet positions
        buffer_length (float): buffer length
        lx (float): x-length of the design domain
        ly (float): y-length of the design domain
    """
    # cycle over the outlets
    com = gmsh.model.occ.getCenterOfMass(line_dim, line_tag)
    mass = gmsh.model.occ.getMass(line_dim, line_tag)
    for outlet_position in outlet_positions:
        x1, y1, x2, y2 = outlet_position
        if x1 == 0 and x2 == 0:  # this outlet is on the left edge
            com_outlet = [-buffer_length, (y1 + y2) / 2.0]
        elif y1 == 0 and y2 == 0:  # this outlet is on the bottom edge
            com_outlet = [(x1 + x2) / 2.0, -buffer_length]
        elif y1 == ly and y2 == ly:  # this outlet is on the top edge
            com_outlet = [(x1 + x2) / 2.0, ly + buffer_length]
        elif x1 == lx and x2 == lx:  # this outlet is on the right edge
            com_outlet = [lx + buffer_length, (y1 + y2) / 2.0]

        mass_outlet = distance(x1, y1, x2, y2)

        # add outlet to the outlets IDs
        if (
            np.isclose(com[0], com_outlet[0])
            and np.isclose(com[1], com_outlet[1])
            and np.isclose(mass, mass_outlet)
            and line_tag not in outlet
        ):
            outlet.append(line_tag)


def get_non_walls(
    non_walls: list, lines: list, inlet_positions: list, outlet_positions: list
):
    """Get invalid walls (interfaces)

    Args:
        non_walls (list): list to be updated with invalid walls
        lines (list): list of all 1d lines
        inlet_positions (list): list with all the inlet positions
        outlet_positions (list): list with all the outlet positions
    """
    # cycle over the lines to catch the lines which are boundaries of the design domain but are not
    # boundaries of the actual domain because of inlets and outlets branches
    for line_dim, line_tag in lines:
        # get center of mass and length of the edge
        com = gmsh.model.occ.getCenterOfMass(line_dim, line_tag)
        mass = gmsh.model.occ.getMass(line_dim, line_tag)

        # get same quantities for the input data of inlets
        for inlet_position in inlet_positions:
            x1, y1, x2, y2 = inlet_position
            com_inlet = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
            mass_inlet = distance(x1, y1, x2, y2)

            # add the current line to the non_walls if all the data coincides
            if (
                np.isclose(com[0], com_inlet[0])
                and np.isclose(com[1], com_inlet[1])
                and np.isclose(mass, mass_inlet)
                and line_tag not in non_walls
            ):
                non_walls.append(line_tag)

        # get same quantities for the input data of outlets
        for outlet_position in outlet_positions:
            x1, y1, x2, y2 = outlet_position
            com_outlet = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
            mass_outlet = distance(x1, y1, x2, y2)

            # add the current line to the non_walls if all the data coincides
            if (
                np.isclose(com[0], com_outlet[0])
                and np.isclose(com[1], com_outlet[1])
                and np.isclose(mass, mass_outlet)
                and line_tag not in non_walls
            ):
                non_walls.append(line_tag)


def create_domain(
    inlet_positions: list[tuple],
    outlet_positions: list[tuple],
    buffer_length: float,
    lx: float,
    ly: float,
    meshfile: str,
    DIVISIONS: int,
    mesh_size: float = None,
):
    r"""
    Create domain and save to a .msh file for further use in Firedrake.

    Args:
        inlet_positions (list): positions of the inlet branches of the type (x1, y1, x2, y2) where
                                (x1, y1) are the coordinates of the first node componing the inlet edge and
                                (x2, y2) are the coordinates of the second node componing the edge. It is required
                                that either x1 == x2 or y1 == y2 and that either x1 \in {0, lx} or x2 \in {0, ly},
                                meaning that the passed coordinates should be points on the boundary of the
                                design domain [0, lx] \times [0, ly]
        outlet_positions (list): positions of the outlet branches of the type (x1, y1, x2, y2) where (x1, y1) are the
                                 coordinates of the first node componing the inlet edge and
                                 (x2, y2) are the coordinates of the second node componing the edge. It is required
                                 that either x1 == x2 or y1 == y2 and that either x1 \in {0, lx} or x2 \in {0, ly},
                                 meaning that the passed coordinates should be points on the boundary of the
                                 design domain [0, lx] \times [0, ly]
        buffer_length (float): length of the inlet and outlet branches, meaning the lateral deviation from the
                               coordinates passed to inlet_positions and outlet_positions
        lx (float): x-length of the design domain
        ly (float): y-length of the design domain
        DIVISIONS (int): number of divisions to use in the mesh
        meshfile (str): .msh file to be imported as a mesh, this is created before the start of the optimization
                        by the method create_domain in utils/create_domain.py
        mesh_size (float): characteristic length of the mesh. Defaults to None, in which case it is set to
                           min(lx, ly) / DIVISIONS
    Return:
        inlet (dict): a dictionary of the inlets with key given by the ID of the inlet to use in Firedrake
                      and pair a tuple given by (sign of the outward pressure direction depending on the position
                      of the inlet, component of the test function which is not cancelled out in the weak formulation)
        outlet (list): a list with the IDs of the outlets to use in Firedrake. Defaults to 1
                       (unchanged with respect to the input variable NON_DESIGN)
        DESIGN (int, optional): ID to utilize for the design domain, namely the inner rectangle [0, lx] \times [0, ly].
                                Defaults to 3
        NON_DESIGN (int, optional): ID to utilize for the non design domain defaults to 1
    """
    DESIGN: int = 3
    NON_DESIGN: int = 1
    if rank == 0:
        mesh_time = time.time()

        gmsh.initialize(sys.argv)

        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 8)

        gmsh.model.add("domain_model")

        if mesh_size is None:
            mesh_size = min(lx, ly) / DIVISIONS

        # Create main domain
        domain = gmsh.model.occ.addRectangle(0, 0, 0, lx, ly)

        inlets, outlets = create_inlets_outlets(
            inlet_positions, outlet_positions, buffer_length, lx, ly
        )

        # Fragment the domain and inlets/outlets
        gmsh.model.occ.fragment([(2, domain)], [(2, i) for i in inlets + outlets])

        # Synchronize model with OCC to finalize the changes
        gmsh.model.occ.synchronize()

        surfaces = gmsh.model.occ.getEntities(dim=2)
        fluid = [surface_tag for _, surface_tag in surfaces]
        non_design_domain = [
            surface_tag for _, surface_tag in surfaces if surface_tag != domain
        ]
        domain = [surface_tag for _, surface_tag in surfaces if surface_tag == domain]
        gmsh.model.addPhysicalGroup(
            2, non_design_domain, NON_DESIGN, "non_design_domain"
        )
        gmsh.model.addPhysicalGroup(2, domain, DESIGN, "domain")
        gmsh.model.addPhysicalGroup(2, fluid, -1, "fluid")

        lines = gmsh.model.occ.getEntities(dim=1)

        inlet, outlet, non_walls = boundary_splitting(
            inlet_positions, outlet_positions, buffer_length, lx, ly
        )

        # assert that a correct number of lines was captured in the above checks, raise an error otherwise
        assert len(non_walls) == len(inlet_positions) + len(outlet_positions), (
            f"During the above checks {len(non_walls)} edges were captured as invalid walls"
            f" but {len(inlet_positions) + len(outlet_positions)} total branches were given in input."
        )

        # get walls as all lines which are not inlets nor outlets nor invalid walls
        walls = [
            line_tag
            for _, line_tag in lines
            if line_tag not in inlet
            and line_tag not in outlet
            and line_tag not in non_walls
        ]

        # create physical group for each inlet
        for _in, key in enumerate(inlet, start=1):
            gmsh.model.addPhysicalGroup(1, [key], key, "inlet " + str(_in))

        # create physical group for each outlet
        for out, key in enumerate(outlet, start=1):
            gmsh.model.addPhysicalGroup(1, [key], key, "outlet " + str(out))

        # create unique "outlet", "walls" and "non_walls" physical groups, the latter mostly for checks
        gmsh.model.addPhysicalGroup(1, walls, 3, "walls")
        gmsh.model.addPhysicalGroup(1, non_walls, 100, "non_walls")

        # Generate 2D mesh
        gmsh.model.occ.synchronize()

        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

        gmsh.model.mesh.generate(2)

        # Save the model
        global_print(f"Saving mesh to {config.meshdir + meshfile}")
        gmsh.write(config.meshdir + meshfile)

        geofile = meshfile.split(".")[0] + ".geo_unrolled"
        global_print(f"Saving geo file to {config.meshdir + geofile}")
        gmsh.write(config.meshdir + geofile)

        gmsh.finalize()

        global_print(
            f"{Fore.GREEN}Domain generation .............. {time.time()-mesh_time : 2.2f} s{Fore.RESET}"
        )
    else:
        inlet = None
        outlet = None

    inlet = COMM_WORLD.bcast(inlet, root=0)
    outlet = COMM_WORLD.bcast(outlet, root=0)

    return inlet, outlet, DESIGN, NON_DESIGN


def set_ordering(
    x: np.array,
    y: np.array,
    transfinite: bool,
    divisions: int,
    subfolder: str = None,
):
    r"""Set ordering depending on type of mesh generated, save the ordering to pickle

    The ordering is based of how we read a book, i.e. from left to right and from top to bottom.

    E.g. the ordering of a 3x3 mesh is:

                   0 ---------- 1 ---------- 2
                   |            |            |
                   |            |            |
                   |            |            |
    y              3 ---------- 4 ---------- 5
    ^              |            |            |
    |              |            |            |
    |              |            |            |
    +-------> x    6 ---------- 7 ---------- 9

    NOTE: np.lexsort([x, -y]) is the same as the other method but only be used for FNOs and for the local x & y
    coordinates (i.e. transfinite and subfolder is not None). Sometimes it is not as reliable! (rounding errors?)

    Args:
        x (np.array): x (n,) coordinates of the mesh all the n points
        y (np.array): y (n,) coordinates of the mesh all the n points
        transfinite (bool): True if transfinite mesh, False otherwise
        divisions (int): number of divisions in the transfinite mesh
        subfolder (str): subfolder where to save the computed ordering

    Returns:
        ordering (np.array): ordering for Fourier series or simple ordering
    """
    # if transfinite and nproc > 1:
    #     raise RuntimeError("FNO dataset generation in parallel is not supported")
    # print("rank", rank, "size", nproc)
    # print("x", x)
    # print("y", y)
    # divisions =

    if transfinite:
        ordering = np.lexsort([max(y) - y])

        print("ordering", ordering)

        ordering = ordering.reshape((divisions, int(y.shape[0] / divisions)))

        print("ordering", ordering)

        ordering = np.flip(ordering, axis=1)
        print("ordering", ordering)
        for j in range(divisions):
            ordering[j, :] = sorted(ordering[j, :], key=lambda it: x[it])
        ordering = ordering.reshape(x.shape[0])
        print("ordering", ordering)

        if subfolder is not None:
            with open(f"{config.resultdir}{subfolder}ordering.pkl", "wb") as handle:
                pickle.dump({"FO": ordering}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        ordering = np.arange(x.shape[0])

    return ordering


def sort_points(combined_coords):
    # Sort the points using the custom sorting key
    sorted_indices = np.lexsort((combined_coords[:, 0], -combined_coords[:, 1]))
    return sorted_indices


def set_ordering_2(
    x: np.array, y: np.array, transfinite: bool, divisions: int, subfolder: str = None
):
    if transfinite:
        combined_coords = np.vstack((x, y)).T

        ordering = np.lexsort((combined_coords[:, 0], -combined_coords[:, 1]))
        if subfolder is not None:
            with open(f"{config.resultdir}{subfolder}ordering.pkl", "wb") as handle:
                pickle.dump({"FO": ordering}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        ordering = np.arange(x.shape[0])
        print("x", x)
        print("y", y)
        print("ordering normal", ordering)

    return ordering


def write_data(
    dofs_h5: fd.HDF5File,
    inputs: dict = None,
    outputs: dict = None,
    ordering: np.array = None,
    current_it: int = 0,
):
    """Write data to HDF5 file, possibly after Fourier series reordering of the dofs
       This is done by Firedrake in parallel so no need for COMM_WORLD.gather(..., root=0)

    Args:
        dofs_h5 (fd.HDF5File): firedrake file to write data to
        inputs (dict): dictionary for input variables
        outputs (dict): dictionary for output variables after ordering is applied
        ordering (np.array): ordering for Fourier series or simple ordering
        current_it (int, optional): current iterate for naming of the variables
    """
    if inputs is None:
        inputs = {}
    if outputs is None:
        outputs = {}
    assert inputs.keys() == outputs.keys()

    for key in outputs:
        outputs[key].dat.data[:] = inputs[key].dat.data[ordering]
        dofs_h5.write(outputs[key], f"{key}_{current_it}")


def set_saving_files(mesh, subfolder: str = None):
    r"""Set saving files for output"""
    global_print(f"{Fore.BLUE}Saving results in {subfolder}")

    test = fd.HDF5File("test", "w", mesh.comm)


def main():

    buffer_length = 0.1
    lx = 1.0
    ly = 1.0
    DIVISIONS = int(3)
    meshfile = "test.msh"

    transfinite = True

    meshfile2 = "test2.msh"

    inlet, outlet, DESIGN, NON_DESIGN = create_transfinite_domain(
        buffer_length, lx, ly, DIVISIONS, meshfile
    )
    # print(inlet)

    # visualize_mesh_transfinite(config.meshdir + meshfile)

    mesh = fd.Mesh(config.meshdir + meshfile)

    # get dummy function for coordinates just to export it easily
    X, Y = (
        mesh.coordinates.dat.data[:, 0],
        mesh.coordinates.dat.data[:, 1],
    )

    # print("Y", Y)
    print("ordering_2", set_ordering_2(X, Y, transfinite, DIVISIONS))
    # set_ordering(X, Y, transfinite, DIVISIONS)

    inlet_positions = [(0, 0, 0, ly)]
    outlet_positions = [(lx, 0, lx, ly)]

    transfinite = False

    inlet, outlet, DESIGN, NON_DESIGN = create_domain(
        inlet_positions, outlet_positions, buffer_length, lx, ly, meshfile2, DIVISIONS
    )
    mesh = fd.Mesh(config.meshdir + meshfile2)

    # get dummy function for coordinates just to export it easily
    X, Y = (
        mesh.coordinates.dat.data[:, 0],
        mesh.coordinates.dat.data[:, 1],
    )

    set_ordering_2(X, Y, transfinite, DIVISIONS)

    visualize_mesh(config.meshdir + meshfile2)


if __name__ == "__main__":
    main()
