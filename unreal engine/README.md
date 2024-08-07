create a film/video blank project with raytracing enabled


add USD Importer plugin
add the downloaded folder containing the .uasset files in the content folder of the project file.
Restart.

in project settings -> engine -> rendering change the default anti aliasing method to MSAA to remove flickering


add a "object" actor tag and activate the CustomDepth option to the DepthLight mesh and any potential hidden shadow catching mesh

add the DepthLight custom blueprint camera to the scene