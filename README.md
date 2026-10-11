# Lake Beulah Residence - 1:75 take-apart model

A four-colour, take-apart "dollhouse" model of the Lake Beulah Residence, generated from the architect's plan set and sliced for a Bambu Lab H2C.

## Renders

![From the lake](renders/01%20Lake%20side.png)

| | |
|---|---|
| ![Entry side](renders/02%20Entry%20side.png) | ![South-east](renders/03%20South-east.png) |
| ![Roofs off](renders/05%20Roofs%20off.png) | ![Exploded](renders/07%20Exploded.png) |

![All pieces](renders/09%20All%20pieces.png)

More views are in `renders/`, with one picture per piece in `renders/pieces/`. They are made by `code/make_renders.py`.

## What's here

| Folder / file | Contents |
|---|---|
| `print/` | One Bambu Studio project per piece (18 pieces), four filaments assigned |
| `sliced/` | All 18 pieces on 18 plates: the project and the sliced `.gcode.3mf` |
| `Assembled model - reference only.3mf` | Everything put together, for viewing only |
| `STL parts (fallback).zip` | Per-colour STLs of every piece |
| `code/` | The scripts that build the model |
| `renders/` | Rendered views of the model and of each piece |
| `*.png` | Working preview pictures |

## Pieces

- **A1-A6** base tiles (ground, first floor, lower level), joined with dovetails
- **B1-B4** upper-floor trays: main suite, north bedrooms, billiards, golf/bar/showroom
- **C1-C8** lift-off roofs

Pieces locate on keys and hold with 2 mm x 1 mm disc magnets (38 pairs, 76 magnets) in wall pockets.

## Filaments

| Slot | Colour | Used for |
|---|---|---|
| 1 | Green | Ground |
| 2 | Tan | Stone and interior |
| 3 | Dark brown | Siding, roofs, wood, furniture |
| 4 | Blue | Glass |

Sliced with 0.4 mm nozzles, 0.20 mm layers, 10% infill, tree supports: about 5 days 3.5 hours and 3.5 kg of PLA in total.

## Rebuilding

Needs Python with numpy, scipy and Pillow.

```
mkdir -p work out
cp code/inputs/w11.npy code/inputs/w12.npy work/
python3 code/build75.py work
python3 code/export75.py work out code
python3 code/multiplate.py "out/all plates.3mf"
```

`build75.py` holds the whole house as code (plan coordinates in feet, x east, y south). `inputs/w11.npy` and `w12.npy` are the interior wall masks traced from the first- and second-floor plans; `intwalls.py`, `extract.py` and `tile.py` are the helpers that made them from the plan PDFs.

The architect's drawings are not included in this repository.
