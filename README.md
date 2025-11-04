# Object Detection System

## Task Overview
Develop an application that recognizes the shape, color and position of objects captured by a video camera. The system must automatically detect changes in the scenario. 

## Setup Requirements
- Objects are placed on an 18x18cm flat surface (e.g. colored paper)
- Positions are output as coordinates in a unit square with corners at (0,0) and (1,1)
- Positions must be verifiable (e.g. through markings on the surface)

## Object Set Requirements
1. Each object must be solid-colored (one color only)
2. Minimum of 4 different object shapes (e.g. cube, pyramid, cylinder, sphere)
3. Minimum of 4 different object colors (e.g. red, green, blue, yellow)
4. Each shape must exist in at least 2 colors
5. At least 4 objects must exist in duplicate (or more)

### Example Object Set
 - 2 red cubes
 - 2 yellow cubes
 - 2 blue pyramids
 - 2 yellow pyramids
 - 1 red cylinder
 - 1 green cylinder
 - 1 green sphere
 - 1 blue sphere

## Implementation Notes
- 3D objects are preferred
- If using 2D objects (paper shapes, coins, flat LEGO pieces), the software must handle overlapping
- For 3D objects, overlapping is excluded during placement