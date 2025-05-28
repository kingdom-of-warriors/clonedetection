from transformers import RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification
import os

def download_roberta_model(model_name="roberta-base", save_dir="./models/roberta-base"):
    """
    下载RoBERTa模型到本地
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"正在下载 {model_name} 的配置文件...")
    config = RobertaConfig.from_pretrained(model_name)
    config.save_pretrained(save_dir)
    
    print(f"正在下载 {model_name} 的分词器...")
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_dir)
    
    print(f"正在下载 {model_name} 的模型...")
    model = RobertaForSequenceClassification.from_pretrained(model_name)
    model.save_pretrained(save_dir)
    
    print(f"模型已保存到: {save_dir}")

if __name__ == "__main__":
    # 根据您的需要修改模型名称
    download_roberta_model("microsoft/graphcodebert-base", "./models/graphcodebert-base")
    # 如果需要其他模型，可以继续添加
    # download_roberta_model("microsoft/codebert-base", "./models/codebert-base")