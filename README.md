# Multi Transformer with FPN as image feature extractor

File description:
- convert_dataset.py to convert X-RAY images and annotations to COCO dataset format
- train.py to train the network

Current best result:
At Epoch 67 - ckpt-30 - eval in 100 data - BEAM_SIZE = 8
Bleu_1: 0.410
Bleu_2: 0.280
Bleu_3: 0.208
Bleu_4: 0.163
computing METEOR score...
METEOR: 0.184
computing Rouge score...
ROUGE_L: 0.347
computing CIDEr score...
CIDEr: 0.844