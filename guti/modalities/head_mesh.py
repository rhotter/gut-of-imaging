import os
import gmsh
import meshio
from jax_fem.utils import get_meshio_cell_type

# Define a function to create a sphere mesh using gmsh.
def get_sphere(radius, mesh_size, data_dir, ele_type='TET4'):
    """
    Generate a 3D tetrahedral mesh of a sphere with given radius and mesh size.
    The mesh is saved to a subfolder 'msh' in data_dir.
    """
    cell_type = get_meshio_cell_type(ele_type)
    msh_dir = os.path.join(data_dir, 'msh')
    os.makedirs(msh_dir, exist_ok=True)
    msh_file = os.path.join(msh_dir, 'sphere.msh')

    gmsh.initialize()
    gmsh.model.add("sphere")
    # Create a sphere centered at (0, 0, 0)
    gmsh.model.occ.addSphere(0, 0, 0, radius)
    gmsh.model.occ.synchronize()
    
    # Set a global mesh size for the sphere
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    
    gmsh.model.mesh.generate(3)
    gmsh.write(msh_file)
    gmsh.finalize()
    
    mesh = meshio.read(msh_file)
    # Extract the tetrahedral elements from the mesh.
    cells = mesh.cells_dict[cell_type]
    return meshio.Mesh(points=mesh.points, cells={cell_type: cells})


def get_fused_spheres(outer_radius, inner_radius, inner_center, outer_mesh_size, inner_mesh_size, data_dir, ele_type='TET4'):
    """
    Generate a 3D tetrahedral mesh of two fused spheres with different mesh sizes.
    
    Args:
        outer_radius (float): Radius of the larger sphere
        inner_radius (float): Radius of the smaller sphere
        inner_center (array-like): (x, y, z) coordinates of the smaller sphere's center
        outer_mesh_size (float): Mesh size for the outer sphere
        inner_mesh_size (float): Mesh size for inner sphere
        data_dir (str): Directory to save the mesh file
        ele_type (str): Element type (default: 'TET4')
    """
    cell_type = get_meshio_cell_type(ele_type)
    msh_dir = os.path.join(data_dir, 'msh')
    os.makedirs(msh_dir, exist_ok=True)
    msh_file = os.path.join(msh_dir, 'fused_spheres.msh')

    gmsh.initialize()
    gmsh.model.add("fused_spheres")
    
    # Create the outer sphere centered at origin
    outer_sphere = gmsh.model.occ.addSphere(0, 0, 0, outer_radius)
    
    # Create the inner sphere at specified location
    x, y, z = inner_center
    inner_sphere = gmsh.model.occ.addSphere(x, y, z, inner_radius)
    
    # Fuse the spheres together
    fused, _ = gmsh.model.occ.fuse([(3, outer_sphere)], [(3, inner_sphere)])
    gmsh.model.occ.synchronize()
    
    # Create a mathematical field for mesh size
    gmsh.model.mesh.field.add("MathEval", 1)
    size_formula = f"{inner_mesh_size} + ({outer_mesh_size}-{inner_mesh_size})*sqrt((x-{x})^2 + (y-{y})^2 + (z-{z})^2)/{outer_radius}"
    gmsh.model.mesh.field.setString(1, "F", size_formula)
    
    # Use the mathematical field as background mesh
    gmsh.model.mesh.field.setAsBackgroundMesh(1)
    
    # Disable other mesh size sources
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    
    # Additional meshing options for better quality
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay algorithm
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    
    # Generate mesh
    gmsh.model.mesh.generate(3)
    gmsh.write(msh_file)
    gmsh.finalize()
    
    mesh = meshio.read(msh_file)
    cells = mesh.cells_dict[cell_type]
    return meshio.Mesh(points=mesh.points, cells={cell_type: cells})

# Define a function to create a five-layer concentric sphere mesh using gmsh.
def get_multilayer_sphere(outer_radii, mesh_size, data_dir, ele_type='TET4'):
    """
    Generate a 3D tetrahedral mesh of a five-layer concentric sphere with given outer radius and mesh size.
    The layers represent (from outside in):
       - Scalp:      between r1 and r2
       - Skull:      between r2 and r3
       - CSF:        between r3 and r4
       - Grey Matter:between r4 and r5
       - White Matter:inside r5
    The mesh is saved to a subfolder 'msh' in data_dir.
    """
    gmsh.initialize()
    gmsh.model.add("multilayer_sphere")
    
    # Define radii for each interface.
    r1 = outer_radii[0]      # outermost surface (scalp outer)
    r2 = outer_radii[1]      # scalp inner / skull outer
    r3 = outer_radii[2]      # skull inner / CSF outer
    r4 = outer_radii[3]      # CSF inner / grey matter outer
    r5 = outer_radii[4]      # grey matter inner / white matter outer
    
    # Create six spheres (volumes) with these radii.
    s1 = gmsh.model.occ.addSphere(0, 0, 0, r1)
    s2 = gmsh.model.occ.addSphere(0, 0, 0, r2)
    s3 = gmsh.model.occ.addSphere(0, 0, 0, r3)
    s4 = gmsh.model.occ.addSphere(0, 0, 0, r4)
    s5 = gmsh.model.occ.addSphere(0, 0, 0, r5)
    # For white matter, we simply use s5 as the inner boundary.
    gmsh.model.occ.synchronize()
    
    # Create the layer volumes via Boolean cuts.
    # Use removeTool=False to preserve the inner sphere for later cuts.
    scalp_vol, _ = gmsh.model.occ.cut([(3, s1)], [(3, s2)], removeTool=False)
    skull_vol, _ = gmsh.model.occ.cut([(3, s2)], [(3, s3)], removeTool=False)
    csf_vol, _   = gmsh.model.occ.cut([(3, s3)], [(3, s4)], removeTool=False)
    grey_vol, _  = gmsh.model.occ.cut([(3, s4)], [(3, s5)], removeTool=False)
    # White matter is simply the innermost sphere.
    white_vol = [(3, s5)]
    
    gmsh.model.occ.synchronize()
    
    # Collect the volume tags from each layer.
    all_vol_tags = []
    for vol in scalp_vol:
        all_vol_tags.append(vol[1])
    for vol in skull_vol:
        all_vol_tags.append(vol[1])
    for vol in csf_vol:
        all_vol_tags.append(vol[1])
    for vol in grey_vol:
        all_vol_tags.append(vol[1])
    for vol in white_vol:
        all_vol_tags.append(vol[1])
    
    # Define a physical group for the entire multi-layer domain.
    gmsh.model.addPhysicalGroup(3, all_vol_tags, tag=1)
    
    # Set the mesh size.
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    
    gmsh.model.mesh.generate(3)
    msh_dir = os.path.join(data_dir, 'msh')
    os.makedirs(msh_dir, exist_ok=True)
    msh_file = os.path.join(msh_dir, 'multilayer_sphere.msh')
    gmsh.write(msh_file)
    gmsh.finalize()
    
    mesh = meshio.read(msh_file)
    cell_type = get_meshio_cell_type(ele_type)
    cells = mesh.cells_dict[cell_type]
    return meshio.Mesh(points=mesh.points, cells={cell_type: cells})