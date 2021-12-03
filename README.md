# AutoSharded_Transformer_Based_on_PyTorch
4 nodes deployment:
CUDA_VISIBLE_DEVICES=4,5,6,7 python Transformer_AutoShard_Test.py
result(max mem set to be 10000):
![image](https://user-images.githubusercontent.com/85312798/144601846-0486f9ce-1569-47e7-892c-b6a3411cc941.png)
50%+ speed up relative to FSDP

8 nodes deployment:
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Transformer_AutoShard_Test.py
result(max mem set to be 10000):
![image](https://user-images.githubusercontent.com/85312798/144602012-da46f4d7-5c6b-408e-8576-30e776f279b1.png)
50%+ speed up relative to FSDP
