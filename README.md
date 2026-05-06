This code is intended to load in localizations from the raw MERFISH imaging cycles and decode the mRNA molecules.

### Data input

Briefly, __mermake__ package (not included here) was used on the raw 3D-zstack images to get a set of local maxima (with sub-pixel fitting).
This was performed for each field of view (i.e. Conv_zscan1__434), for each imaging cycle (i.e. H1_MER_set1) and for each color (i.e. col0).
Compressed .npz files store the fitted data. Example data across all imaging cycles and for all colors is provided for one field of view (Conv_zscan1__434).

Specifically: Data\Conv_zscan1__434--H2_MER_set1--col2__Xhfits.npz contains the local maxima within the 3D image aquired for field of view Conv_zscan1__434, 
during cycle 2 of hybridization and for color 2 (the cy3 channel). 

The fitted data is store as a numpy array (called Xh) of size N molecules x 8. 
The 8 entries represent: z, x , y (positions of local maxima in camera pixels), local_background_brightness, correlation_with_PSF_predeconvolution, brightness_predeconvolution, correlation_with_PSF_afterdeconvolution, brightness_afterdeconvolution.

For each imaging cycle there are typically 3 colors of signal (col0 - 750 channel, col1 - cy5 channel, col2 -cy3 channel) and one additiona channel for dapi. 
There is a file for each cycle containing "dapiFeatures"  in its name. 
For instance Data\Conv_zscan1__434--H2_MER_set1--dapiFeatures__Xhfits.npz file contains local maxima (Xh_plus) and local minima (Xh_minus) within the dapi channel of imaging cycle 2.
These are used for registration acros imaging cyles.

### Decoding

All the decoding functions are contained in __fitfov.py__ file.
An example use case of how to decode the example field of view(__Conv_zscan1__434__ in Data folder) is contained in __ExampleUse.ipynb__

The main function for decoding is:
main(__FOV__,__path_to_fitted_data__,__path_to_auxiliary_helper_files__,verbose=True)

This functions runs the following steps:

### Step 1 imaging cyles registration

The imaging cyles are aligned to a reference cycle (typically cycle 1) based on the dapi features (local maxima and local minimima in the nuclear signal).

First the dapi fetures - local maxima and local minima (__Xh_plus__ and __Xh_minus__) - of the reference cyle is loaded in from __Data\Conv_zscan1__434--H1_MER_set1--dapiFeatures__Xhfits.npz__.

Itterating through imaging cycles, the local maxima and minima of dapi are loaded for each current cycle from __Data\Conv_zscan1__434--H2_MER_set1--dapiFeatures__Xhfits.npz__.

Function __get_best_translation_pointsT(fov,htag,htagref,save_folder,save_folder_reference,set_='',resc=5,th=0)__ takes in the following mandatory variables: name of the field of view (i.e. Conv_zscan1__434) the current name of the cycle (i.e. H2_MER_set1), the name of the reference cycle (i.e. H1_MER_set1 - the first cycle),
the name of the folder containing the dapiFeatures of the current cycle and the reference cycle (save_folder,save_folder_reference).

This function returns variable __drft__ which is a list of 5 elements: the z,x,y final drift correction (tzxy_final), the drift calculated only based on local maxima (tzxy_plus), the drift calculated only based on local minima (tzxy_minus), the number of local maxima dapi features aligned after drift,  the number of local minima dapi features aligned after drift. 

A list acorss imaging cycles containing [[drft_cycle1, drft_cycle2,...],[name_cycle1,name_cycle2,...],fov_name,name_reference_cyle] is saved in the same folder as the dapiFeatures with name: __driftNew_Conv_zscan1__434--.pkl__

Briefly, tzxy_plus is computed by first transforming Xh_plus[:,:3] for the current cycle and Xh_plus[:,:3] for the reference cycle into 3D images (im_dapi_cycle, im_dapi_reference) rescaled by __resc__ (typically 5 times).

The best translation of images im_dapi_cycle, im_dapi_reference is computated based on fft correlation. This is then used to determine nearest neigbors from Xh_plus_current_cyle[:,:3] to Xh_plus_reference[:,:3] and refine the alignment to subpixel.

A scaled average between tzxy_plus and tzxy_minus based on number of references found for each is used to computed the final drift tzxy_final.

To bring localizations in cycle N to the reference cycle the following transformation is applied: Xh_cycleN[:,:3] = Xh_cycleN[:,:3]+tzxy_final


### Step 2 Loading in the localizations across all cycles and colors for the target FOV

For each field of view the fitted data for each cycle and each color is loaded in.

This is perfomed by function: get_XH(dec,fov,ncols=3,th_h=0,
                   color_fl=helper_folder+os.sep+'color_correction.pkl',
                   medH_fl=helper_folder+os.sep+'medHBRBB.npz')

The following corrections are applied to each loaded fits (Xh):










