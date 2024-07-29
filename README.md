# TSPM
Official repository for "Boosting Audio Visual Question Answering via Key Semantic-Aware Cues" in ACM MM 2024.

[Guangyao Li](https://ayameyao.github.io/), [Henghui Du](),  [Di Hu](https://dtaoo.github.io/index.html)



## Requirements

```python
python3.6 +
pytorch1.6.0
tensorboardX
ffmpeg
numpy
```



## Usage

1. **Clone this repo**

   ```python
   git clone https://github.com/GeWu-Lab/TSPM.git
   ```

2. **Download data**

   MUSIC-AVQA: https://gewu-lab.github.io/MUSIC-AVQA/

   AVQA: http://mn.cs.tsinghua.edu.cn/avqa/

3. **Feature extraction**

   ```python
   cd feat_script/extract_clip_feat
   python extract_qst_ViT-L14@336px.py
   python extract_qaPrompt_ViT-L14@336px
   python extract_token-level_feat
   python extract_frames_ViT-L14@336px
   ```

4. Training

   ```python
   python -u main_train.py --Temp_Selection --top_k 10 \
   						--Spatio_Perception \
   						--batch-size 64 --epochs 30 --lr 1e-4 \
   						--num_workers 12 --gpu 0,1 \
   						--checkpoint TSPM \
   						--model_save_dir models
   ```

5. Testing

   ```python
   python -u main_test.py --Temp_Selection --top_k 10 \
   			  --Spatio_Perception \
   			  --batch-size 1 --gpu 1 \
   			  --checkpoint TSPM \
   			  --model_save_dir models \
   			  --result_dir results
   ```




## Citation

If you find this work useful, please consider citing it.

```
coming soon!
```



## Acknowledgement

This research was supported by Public Computing Cloud, Renmin University of China.
