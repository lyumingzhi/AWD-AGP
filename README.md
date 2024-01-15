# The Official Implementation of Paper _Adversarial Attack for RobustWatermark Protection Against Inpainting-based and BlindWatermark Removers_

## Structure
* Item inpainting
    * Item AWD-AGP  
        * Item source_models (Corresponding watermark removers should be firstly installed by users and adapted to the APIs suggested as the following documents.)
            * Item WDnet.py: API of WDnet (_WDNet: Watermark-Decomposition Network for Visible Watermark Removal_)
            * Item Mat.py: API of Mat (_MAT: Mask-Aware Transformer for Large Hole Image Inpainting_)
            * Item FcF.py: API of FcF (_Keys to Better Image Inpainting: Structure and Texture Go Hand in Hand_)
            * Item SLBRnet.py: API of SLBRnet (_Visible Watermark Removal via Self-calibrated Localization and Background Refinement_)
            * Item Generative.py: API of Generative-Inpainting-pytorch (_Generative Image Inpainting with Contextual Attention_)
            * Item Crfill.py: API of Crfillnet (_CR-Fill: Generative Image Inpainting with Auxiliary Contextual Reconstruction_)

        * Item options.py: args shared by different scripts.
        * Item official_dataset.py: the dataset package.
        * Item official_surrogate_generate_AE.py: the main script to generate transferable adversarial examples.
            > Example Command: 
            ```js 
            python -m inpainting.ensemble.official_surrogate_generate_AE --data places2 --algorithm random_logo_alpha --algorithm norm_combine  --lossType perceptual_loss --InputImg_dir Dir_of_input_images  --output_dir  Output_dir_of_adversarial_examples  --target_model Target_model (choose from ['Crfillnet', 'Gennet', 'FcFnet', 'Matnet', 'WDModel', 'SLBRnet'])   --get_logo --attach_logo --algorithm attribution_attack_for_multi_task3  --get_logo --attach_logo --RPNRefineMask Dir_of_json_file_for_watermark_locations
            ```
        * Item official_Attack.py: the attack package.
        * Item official_transfer_attack_for_wr.py: the main script to do transfer attack on a blind watermark remover with the generated adversarial examples.
            > Example Command: 
            ```js
            python -m inpainting.ensemble.official_transfer_attack_for_wr --dataset places2 --lossType perceptual_loss --InputImg_dir Dir_of_input_images  --output_dir Dir_of_output_of_adversarial_examples_in_transfer_attack --experiment_dir Dir_of_adversarial_examples --target_model Target_model --algorithm optimal_mask_search --get_logo --attach_logo
            ```
        * Item official_transfer_attack.py: the main script to do transfer attack on an inpainting-based remover with the generated adversarial examples.
            > Example Command: 
            ```js
            python -m inpainting.ensemble.official_transfer_attack --dataset places2 --lossType perceptual_loss --InputImg_dir Dir_of_input_images  --output_dir Dir_of_output_of_adversarial_examples_in_transfer_attack --experiment_dir Dir_of_adversarial_examples --target_model Target_model --algorithm optimal_mask_search --get_logo --attach_logo
            ```
        * Item official_evaluate_for_wr.py: the main evaluation script for blind watermark removal results.
            > Example Command:
            ```js
            python -m inpainting.ensemble.official_evaluate_for_wr --target_model Target_model --output_dir Dir_of_output_of_adversarial_examples_in_transfer_attack --InputImg_dir /home1/mingzhi/inpainting/Watermarking  --source_dir Dir_of_input_images --get_logo --algorithm evaluate_rw
            ```
        * Item official_evaluate.py: the main evaluation script for inpainting-based removal results.
            > Example Command: 
            ```js 
            python -m inpainting.ensemble.official_evaluate --target_model Target_model --output_dir Dir_of_output_of_adversarial_examples_in_transfer_attack --InputImg_dir /home1/mingzhi/inpainting/Watermarking  --source_dir Dir_of_input_images --get_logo
            ```
        * Item official_noise_func.py: the noise model package.
        * Item official_utils.py: the utils package.
        * Item official_surrogate_generate_AE_mask_DE_without_noise.py: the main script to generate optimal watermark location.
            > Example Command: 
            ```js 
            python -m inpainting.ensemble.surrogate_generate_AE_mask_DE_for_each_model --dataset places2 --target_models Target_model --lossType perceptual_loss --InputImg_dir Dir_of_input_images --output_dir Dir_of_output_of_adversarial_examples_in_transfer_attack
            ```
        * Item superpixel_fcn: superpixel generation package (https://github.com/fuy34/superpixel_fcn)
    * Item Mat: Mat package, (which users need to install and adapt to the API suggested in the source_models folder. So do following watermark remover packages)
    * Item generative_inpainting
    * Item FcF_inpainting
    * Item crfill
    * Item generative_inpainting_pytorch
    * Item WDNet

![The attack results of AWD-AGP and DWV on WDModel. The images in the first row are watermarked images and the images in the second row are the attack results.]("https://github.com/lyumingzhi/AWD-AGP/blob/main/blind_watermark_removal.jpg")
![The attack results of AWD-AGP and markpainting on FcFnet. The images in the first row are watermarked images and the images in the second row are the attack results.]("https://github.com/lyumingzhi/AWD-AGP/blob/main/inpainting_based_removal.jpg")