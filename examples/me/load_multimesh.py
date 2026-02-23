"""
Script to load any OBJ file containing multiple sub-meshes adaptively.
Can be used with files like thorax.obj or lunglobes.obj.
"""

import argparse
import os
import numpy as np
import meshio
import genesis as gs
import random

def load_submeshes_by_name(obj_file):
    """
    Load an OBJ file and extract individual sub-meshes based on object names.
    Handles vertex normal mismatches gracefully.
    
    Parameters
    ----------
    obj_file : str
        Path to the OBJ file containing multiple named sub-meshes.
    
    Returns
    -------
    submeshes : dict
        Dictionary mapping sub-mesh names to meshio.Mesh objects.
    """
    submeshes = {}
    
    if not os.path.exists(obj_file):
        raise FileNotFoundError(f"File not found: {obj_file}")

    print(f"Parsing {obj_file}...")
    
    # Parse the OBJ file directly to identify object boundaries and extract geometry
    with open(obj_file, 'r') as f:
        lines = f.readlines()
    
    # Collect all vertices globally (OBJ uses global vertex indexing)
    all_vertices = []
    for line in lines:
        if line.startswith('v '):
            parts = line.strip().split()
            v = [float(parts[1]), float(parts[2]), float(parts[3])]
            all_vertices.append(v)
    
    all_vertices = np.array(all_vertices)
    
    # Find object boundaries
    object_ranges = {}
    current_obj_name = "default_object"
    
    # First pass to find object names and ranges
    obj_indices = []
    for i, line in enumerate(lines):
        if line.startswith('o '):
            obj_indices.append((i, line.strip()[2:]))
            
    if not obj_indices:
        # No objects defined, treat whole file as one object
        object_ranges["default_object"] = {'start': 0, 'end': len(lines)}
    else:
        for k in range(len(obj_indices)):
            start_idx, name = obj_indices[k]
            end_idx = obj_indices[k+1][0] if k + 1 < len(obj_indices) else len(lines)
            object_ranges[name] = {'start': start_idx, 'end': end_idx}
        
    lung_lobes = [item for item in object_ranges.keys() if 'lobe' in item]
    lung_lobes_range = {name: object_ranges[name] for name in lung_lobes}
    veins_arteries = [item for item in object_ranges.keys() if 'vein' in item or 'artery' in item]
    veins_arteries_range = {name: object_ranges[name] for name in veins_arteries}

    print(f"Found {len(object_ranges)} objects: {list(object_ranges.keys())}")
    print(f"Found {len(lung_lobes)} lung lobes: {lung_lobes}")

    del object_ranges
    # combine dicts
    object_ranges = {**lung_lobes_range, **veins_arteries_range}

    # Extract faces for each object
    for obj_name, obj_range in object_ranges.items():
        obj_faces_global = []  # Store faces with global vertex indices
        
        for i in range(obj_range['start'], obj_range['end']):
            line = lines[i]
            
            if line.startswith('f '):
                # Parse face - extract vertex indices only
                parts = line.strip().split()[1:]
                face = []
                for part in parts:
                    # Handle v, v/vt, v/vt/vn, v//vn formats
                    vertex_str = part.split('/')[0]
                    if vertex_str:
                        vertex_idx = int(vertex_str) - 1  # OBJ uses 1-based indexing
                        face.append(vertex_idx)
                
                if len(face) >= 3:
                    # Triangulate if necessary (fan triangulation)
                    for j in range(1, len(face) - 1):
                        obj_faces_global.append([face[0], face[j], face[j + 1]])
        
        if obj_faces_global:
            # Extract unique vertices used in this object
            obj_faces_global = np.array(obj_faces_global)
            unique_vertices_idx = np.unique(obj_faces_global.flatten())
            
            # Create mapping from global indices to local indices
            vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertices_idx)}
            
            # Extract vertices and remap faces
            obj_vertices = all_vertices[unique_vertices_idx]
            obj_faces_local = np.array([[vertex_map[v] for v in face] for face in obj_faces_global])
            
            # Create mesh
            submeshes[obj_name] = meshio.Mesh(
                points=obj_vertices,
                cells=[("triangle", obj_faces_local)],
            )
        else:
            print(f"Warning: Object '{obj_name}' has no faces, skipping.")
    
    return submeshes


def save_submesh(mesh, output_file):
    """
    Save a submesh to an OBJ file.
    """
    mesh.write(output_file)
    print(f"Submesh saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Load any multi-mesh OBJ file adaptively into Genesis.")
    parser.add_argument("-f", "--file", type=str, default="/home/flaccagora/Desktop/Genesis/examples/me/test_lobe.obj", help="Path to OBJ file")
    parser.add_argument("-v", "--vis", action="store_true", default=True, help="Show visualization")
    parser.add_argument("-c", "--cpu", action="store_true", default=False, help="Use CPU backend")
    parser.add_argument("-o", "--output", type=str, default="multimesh_render.png", help="Output image filename")
    parser.add_argument("-n", "--num", type=int, default=10000, help="Number of frames to simulate")
    
    args = parser.parse_args()

    print("="*60)
    print("Loading multi-mesh file:", args.file)
    print("="*60)

    try:
        gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32")

        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=4e-3,
                substeps=10,
                gravity=(0, 0, -9.81),   
            ),
            # rigid_options=gs.options.RigidOptions(
            #     gravity=(0, 0, -9.8),
            #     enable_collision=True,
            #     enable_self_collision=False,
            #     enable_adjacent_collision=False,
            #     multiplier_collision_broad_phase=20,
            #     # Prevent the rigid contact solver from being too stiff otherwise this would cause large impulse, especially
            #     # because the simulation timestep must be very small to ensure numerical stability of rigid body dynamics.
            #     constraint_timeconst=0.02,
            # ),

            # mpm_options=gs.options.MPMOptions(
            #     lower_bound=(-2.0, -2.0, -2.0),
            #     upper_bound=(2.0, 2.0, 2.0),
            #     grid_density=64,
            # ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.0, 3.0, 2.0),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
                max_FPS=60,
            ),
            vis_options=gs.options.VisOptions(
                visualize_mpm_boundary=True,
                show_world_frame=True,
            ),
            show_viewer=args.vis,
        )

        # Ground plane
        plane = scene.add_entity(
            morph=gs.morphs.Plane(),
        )

        submeshes = load_submeshes_by_name(args.file)
        
        print(f"Loaded {len(submeshes)} sub-meshes.")
        
        # Add each sub-mesh as a separate entity
        for i, (submesh_name, submesh) in enumerate(submeshes.items()):
            # Save submesh to temporary file
            safe_name = "".join([c if c.isalnum() else "_" for c in submesh_name])
            temp_file = f"/tmp/{safe_name}.obj"
            save_submesh(submesh, temp_file)
            
            # Generate a random color or cycle through a palette
            # Use a deterministic seed based on name to keep colors consistent across runs
            random.seed(submesh_name)
            color = (random.random(), random.random(), random.random(), 0.8)
            
            print(f"Adding entity: {submesh_name}")
            
            try:
                scene.add_entity(
                    material=gs.materials.MPM.Muscle(
                    E=5e3,       # Young's modulus
                    nu=0.4,      # Poisson's ratio
                    rho=500.0,   # Density
                    n_groups=1,                    
                ),
                    morph=gs.morphs.Mesh(
                        file=temp_file,
                        scale=1.0,   # Assuming 1.0 scale, adjust if needed
                        pos=(0, 0, 0.1), # Lift up slightly
                        euler=(0, 0, 0),
                        decimate_aggressiveness=2,  # Adjust decimation aggressiveness as needed
                    ),
                    surface=gs.surfaces.Default(
                        color=color,
                        vis_mode="visual",
                    ),
                )
            except Exception as e:
                print(f"Failed to add entity {submesh_name}: {e}")
                continue

        scene.build()

        if args.vis:
            print("Running simulation...")
            for i in range(args.num):
                scene.step()
        else:
            print("Running headless...")
            for i in range(min(100, args.num)):
                scene.step()
            scene.render(save_to_filename=args.output)
            print(f"Saved render to {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
