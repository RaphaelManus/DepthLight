<div align="center">
<h1>DepthLight in Unreal Engine</h1>
</div>

## Steps
#### Step 1:
* Create a __film/video__ blank project with raytracing enabled
* Add the USD Importer plugin to the project
* In _project settings_ → _engine_ → _rendering_ change the default _anti aliasing_ method to _MSAA_ to prevent temporal flickering.

#### Step 2:
* Download the __depthlight__ folder containing the .uasset files and place it in the __content__ folder of the project file.
* In Unreal Engine, replace the background image with yours.
* Import the _usd_ file containing the _DepthLight mesh_ to your project.

#### Step 3:
* Select the _DepthLight mesh_, and any additional mesh you want to hide in the compositing, add an actor tag named `hidden` and activate the `CustomDepth` option.
* Add the DepthLight blueprint camera to the scene. and center it relative to your _DepthLight mesh_.
