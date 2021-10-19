# 5.8.6. Evaluating inForm® Phenotype QC Output for the *AstroPath Pipeline*

1. Go to the Clinical_Specimen_Y folder (Y is number associated with your cohort)
2. Open the ‘Upkeep and Progress’ folder
3. Open the ‘inform_QC’ spreadsheet
   - Running down the first column is a list of all specimens in the cohort
   - The first row lists all the phenotypes that were run for the cohort
4. Go to the QC images for a case
   - \\bkiZZ\Clinical_Specimen_Y\XX1\inform_data\Phenotyped\Results\QA_QC\Phenotype
      - ZZ: Which BKI server your data is stored on
      - Y: Clinical Specimen number associated with your cohort
      - XX: Reassigned Specimen ID
   - Each phenotype is broken up into an individual folder, with up to 20 im3s represented
   - There are seven versions of each image:
   - cell_stamp_mosiacs_pos_neg: shows 20 positive and 20 negative cells from one image with the stain in white, DAPI in blue, and the cell segmentation shown in red
     - White + indicates a cell being phenotyped as positive by the algorithm
   - cell_stamp_mosiacs_pos_neg_no_dapi: same as previous image without DAPI
   - cell_stamp_mosiacs_pos_neg_no_dapi_no_seg: same as previous without cell segmentation
   - cell_stamp_mosiacs_pos_neg_no_seg: same as first image without cell segmentation
   - full_color_expression_image_no_seg: shows whole image with each all markers shown in original pseudocolor
     - useful in determining if background/nonspecific staining is present
   - single_color_expression_image: shows whole image with only the marker of interest shown in white
   - single_color_expression_image_no_seg: same as previous without cell segmentation
5. Flip through the images for the first phenotype
6. Recording QC:
   - If the phenotype and cell segmentation are correct, place an X in the corresponding cell for that case and phenotype
   - If there are issues, record notes in the corresponding cell
   - Up to researcher to determine what level of error they deem acceptable
7. Once finished with QC for a case, place the date in the corresponding cell
8. Save this file
   - Keep the file as a .csv and ignore the pop-up warning the formatting data will be lost
9. See examples of good/bad phenotyping in PPT
