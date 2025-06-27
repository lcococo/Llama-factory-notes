import torch
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import numpy as np
from typing import Optional

# 全局模型和处理器单例
_model = None
_processor = None

def initialize_model(model_dir: str):
    """初始化模型和处理器（单例模式）"""
    global _model, _processor
    if _model is None or _processor is None:
        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        _processor = AutoProcessor.from_pretrained(model_dir)

def get_image_embedding(
    image_path: str, 
    model_dir: str,
    text_prompt: Optional[str] = None,
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    获取图片的embedding表示
    
    Args:
        image_path: 图片文件路径
        model_dir: 模型目录路径
        text_prompt: 可选文本提示，None表示纯视觉特征
        output_path: 可选输出路径，None表示不保存到文件
        
    Returns:
        numpy数组形式的embedding向量
    """
    # 确保模型已初始化
    initialize_model(model_dir)
    
    try:
        # 加载图片
        image = Image.open(image_path).convert("RGB")
        
        # 构建消息
        content = [{"type": "image", "image": f"file:///{image_path}"}]
        if text_prompt:
            content.append({"type": "text", "text": text_prompt})
            
        messages = [{"role": "user", "content": content}]
        
        # 处理输入
        text = _processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        inputs = _processor(
            text=text,
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to("cuda")
        
        # 模型推理
        with torch.no_grad():
            outputs = _model(**inputs, output_hidden_states=True)
            features = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()
            
        # 如果需要保存到文件
        if output_path:
            np.savetxt(output_path, features[0], fmt='%.6f')
            
        return features[0]
        
    except Exception as e:
        raise RuntimeError(f"图片处理失败: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="模型目录路径")
    parser.add_argument("--image_path", type=str, required=True, help="图片文件路径")
    parser.add_argument("--text", type=str, default=None, help="可选文本提示")
    parser.add_argument("--output", type=str, default=None, help="embedding输出文件路径")
    args = parser.parse_args()
    
    embedding = get_image_embedding(
        image_path=args.image_path,
        model_dir=args.model_dir,
        text_prompt=args.text,
        output_path=args.output
    )
    print(f"Embedding shape: {embedding.shape}")
    print(f"Mean: {np.mean(embedding):.4f}")
    print(f"Std: {np.std(embedding):.4f}")
