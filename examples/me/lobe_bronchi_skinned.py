"""
Lobe + Bronchi simulation using gs.morphs.MeshSet.

All lobe and bronchi sub-meshes are loaded as a SINGLE MPM entity via
MeshSet.  Because they share the same MPM grid the bronchi particles are
part of the same velocity field as the lobe — no per-step coupling code is
needed and there is no inter-entity collision instability.

MeshSet.files must be trimesh.Trimesh objects (not file paths).
Each mesh in the set gets its own muscle-group id automatically via
MPMEntity._add_to_solver → set_muscle_group(mesh_set_group_ids).
"""

import argparse
import re
import numpy as np
import trimesh
import genesis as gs


# ---------------------------------------------------------------------------
# OBJ parsing: returns dict { name -> trimesh.Trimesh }
# ---------------------------------------------------------------------------
EXCLUDE_KEYWORDS = [
    r"label", r"text", r"annotation", r"arrow", r"guide",
    r"camera", r"light", r"armature",
    r"\.g$",   # Z-anatomy uses ".g" suffix for label/group empties
    r"\.j$",
]

def compile_patterns(keywords):
    return [re.compile(k, re.IGNORECASE) for k in keywords]

EXCLUDE_RE = compile_patterns(EXCLUDE_KEYWORDS)


def parse_obj_submeshes(obj_file: str) -> dict[str, trimesh.Trimesh]:
    with open(obj_file, "r") as f:
        lines = f.readlines()

    # Collect global vertex list (OBJ uses global indexing)
    all_verts = []
    for line in lines:
        if line.startswith("v "):
            p = line.split()
            all_verts.append([float(p[1]), float(p[2]), float(p[3])])
    all_verts = np.array(all_verts, dtype=np.float64)

    # Find named-object boundaries
    obj_markers = [(i, line.strip()[2:]) for i, line in enumerate(lines)
                   if line.startswith("o ")]
    if not obj_markers:
        obj_markers = [(0, "default")]
    ranges = {}
    for k, (start, name) in enumerate(obj_markers):
        end = obj_markers[k + 1][0] if k + 1 < len(obj_markers) else len(lines)
        
        if any(rex.search(name) for rex in EXCLUDE_RE):
            continue

        ranges[name] = (start, end)

    submeshes = {}
    for name, (start, end) in ranges.items():
        faces_global = []
        for i in range(start, end):
            line = lines[i]
            if not line.startswith("f "):
                continue
            parts = line.split()[1:]
            vids = [int(p.split("/")[0]) - 1 for p in parts]
            for j in range(1, len(vids) - 1):           # fan triangulation
                faces_global.append([vids[0], vids[j], vids[j + 1]])

        if not faces_global:
            continue

        faces_np = np.array(faces_global, dtype=np.int64)
        unique_idx = np.unique(faces_np)
        mapping = {old: new for new, old in enumerate(unique_idx)}
        verts_local = all_verts[unique_idx]
        faces_local = np.vectorize(mapping.get)(faces_np)

        submeshes[name] = trimesh.Trimesh(
            vertices=verts_local,
            faces=faces_local,
            process=False,
        )

    return submeshes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file",
                        default="/home/flaccagora/Desktop/Genesis/examples/me/test_lobe.obj")
    parser.add_argument("-v", "--vis",   action="store_true", default=False)
    parser.add_argument("--cpu",         action="store_true", default=False)
    parser.add_argument("-n", "--num",   type=int, default=10)
    args = parser.parse_args()

    # ---- Parse sub-meshes -------------------------------------------------
    print(f"Parsing {args.file} ...")
    submeshes = parse_obj_submeshes(args.file)
    print(f"  Found objects: {list(submeshes.keys())}")

    lobe_items    = [(n, m) for n, m in submeshes.items() if "lobe"   in n.lower()]
    bronchi_items = [(n, m) for n, m in submeshes.items() if "bronch" in n.lower()]
    everything_else = [(n, m) for n, m in submeshes.items() if "lobe" not in n.lower() and "bronch" not in n.lower()]

    if not lobe_items:
        raise RuntimeError("No objects with 'lobe' in name.")
    if not bronchi_items:
        raise RuntimeError("No objects with 'bronch' in name.")

    print(f"  Lobes   : {[n for n,_ in lobe_items]}")
    print(f"  Bronchi : {[n for n,_ in bronchi_items]}")
    print(f"  Other   : {[n for n,_ in everything_else]}")

    # Put lobes first so group-ids 0..L-1 = lobes, L..L+B-1 = bronchi
    ordered = lobe_items + bronchi_items + everything_else
    tmp_mesh_list   = [m for _, m in ordered]
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32")

    error_meshes = []
    for i, (name, mesh) in enumerate(ordered):
        # if i >= 20:
        #     continue
        print(i, name)
        try:
            mesh_list = [tmp_mesh_list[8], tmp_mesh_list[i]]
                
            n_groups    = len(mesh_list)
            # Shared position / rotation for every sub-mesh
            pos_list    = [(0.0, 0.0, 0.1)] * n_groups
            euler_list  = [(0.0, 0.0, 0.0)] * n_groups

            print(f"{n_groups} mesh groups (0..{len(lobe_items)-1} = lobes, "
                f"{len(lobe_items)}..{n_groups-1} = bronchi)")


            # ---- Genesis scene ----------------------------------------------------

            scene = gs.Scene(
                sim_options=gs.options.SimOptions(dt=3e-3, substeps=10,
                                                gravity=(0, 0, -9.81)),
                viewer_options=gs.options.ViewerOptions(
                    camera_pos=(3.0, 3.0, 2.0),
                    camera_lookat=(0.0, 0.0, 0.5),
                    camera_fov=40,
                    max_FPS=60,
                ),
                vis_options=gs.options.VisOptions(show_world_frame=True),
                show_viewer=args.vis,

            )

            scene.add_entity(morph=gs.morphs.Plane())

            # ---- Single entity — lobe + bronchi share one MPM simulation ----------
            lung = scene.add_entity(
                material=gs.materials.MPM.Muscle(
                    E=5e3,
                    nu=0.4,
                    rho=500.0,
                    n_groups=n_groups,   # one group per sub-mesh
                ),
                morph=gs.morphs.MeshSet(
                    files=mesh_list,     # list of trimesh.Trimesh objects
                    poss=pos_list,
                    eulers=euler_list,
                    scale=1.0,
                ),
                surface=gs.surfaces.Default(vis_mode="visual"),
            )

            scene.build()

            print(f"Lung entity: {lung.n_particles} total particles, "
                f"{n_groups} mesh groups (0..{len(lobe_items)-1} = lobes, "
                f"{len(lobe_items)}..{n_groups-1} = bronchi)")

            # ---- Simulation loop --------------------------------------------------
            print("Simulating...")
            for _ in range(args.num):
                scene.step()
            
            scene.destroy()
            del scene
            del lung
            del mesh_list

        except Exception as e:
            print(f"Error with mesh '{name}': {e}")
            error_meshes.append(name)
            continue
        
    print(f"Meshes with errors: {error_meshes}")    
    print(f" Error Meshes: {len(error_meshes)}")
    print(f"  Lobes   : {len(lobe_items)}")
    print(f"  Bronchi : {len(bronchi_items)}")

if __name__ == "__main__":
    main()
