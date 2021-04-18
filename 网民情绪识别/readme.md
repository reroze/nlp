由于大小限制 模型参数和训练集没有上传

目录组织为：

code

-bert bert模型训练文件

-code 数据处理和其他模型测试

-data 停用词和数据预处理（数据集无法公开）

-result 各种模型训练后的结果

-code2 新的模型训练的结果



结果一览：

•Albert_tiny(less_cleaned_data,batch_size=256): 0.638

•Albert_tiny(less_data, batch_size=256): 0.625

•Albert_base(less_data, batch_size=32): 0.613

•Bert-base(normal_data, batch_size=32):0.585

•Nezha(less_data, batch_size=32):0.49

•Ernie_large(less_data,batch_size=16):0.38