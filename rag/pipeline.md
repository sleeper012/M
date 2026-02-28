```py
"""
使用本地 Qwen2.5-0.5B-Instruct 模型的 RAG Pipeline
"""

import json
import os
import torch
from typing import Dict, Any, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from local_model_config import (
    LOCAL_MODEL_PATH,
    GENERATION_CONFIG,
    CRITIQUE_GEN_CONFIG,
    REWRITE_GEN_CONFIG,
    CLASSIFICATION_CONFIG,
    CATEGORIES,
    CATEGORY_MAPPING,
    CLASSIFICATION_PROMPT,
    SYSTEM_PROMPT,
    CRITIQUE_PROMPT,
    REWRITE_PROMPT,
    STRUCTURED_LAWS_PATH,
    TOP_K,
    extract_and_map_categories,
    get_random_critique_request,
    CRITIQUE
)


class LocalModelPipeline:
    """使用本地模型的完整RAG Pipeline"""
    
    def __init__(self, model_path: str = LOCAL_MODEL_PATH, use_embedding: bool = True, use_vllm: bool = False):
        """
        初始化Pipeline
        
        Args:
            model_path: 本地模型路径
            use_embedding: 是否使用向量化检索（默认True，推荐）
            use_vllm: 是否使用vLLM加速（默认False，如果True则使用vLLM进行推理加速）
        """
        print(f"正在加载本地模型: {model_path}")
        
        # 验证模型路径
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"模型配置文件不存在: {config_path}")
        
        print(f"  模型路径验证通过: {model_path}")
        
        # 加载法律数据和类别（无论使用哪种推理引擎都需要）
        with open(STRUCTURED_LAWS_PATH, 'r', encoding='utf-8') as f:
            self.laws_by_category = json.load(f)
        
        print(f"✓ 已加载法律数据: {sum(len(laws) for laws in self.laws_by_category.values())} 条法规")
        
        self.categories = CATEGORIES
        
        # 初始化向量化检索相关属性（无论使用哪种推理引擎都需要）
        self.use_embedding = use_embedding
        self.embedding_retriever = None
        
        self.use_vllm = use_vllm
        
        if use_vllm:
            # 使用 vLLM 加速推理
            try:
                from vllm import LLM
                print("  使用 vLLM 加速推理...")
                self.llm = LLM(
                    model=model_path,
                    trust_remote_code=True,
                    dtype="bfloat16"
                )
                # vLLM 会自动加载 tokenizer
                self.tokenizer = self.llm.get_tokenizer()
                self.model = None  # vLLM 模式下不使用 transformers 模型
                print(f"✓ vLLM 模型加载成功")
                
                # 初始化向量化检索器（如果启用）
                if self.use_embedding:
                    self._init_embedding_retriever()
            except ImportError:
                print("  ⚠️ vLLM 未安装，回退到 transformers")
                print("  安装 vLLM: pip install vllm")
                self.use_vllm = False
                self._load_transformers_model(model_path)
            except Exception as e:
                print(f"  ⚠️ vLLM 初始化失败: {e}，回退到 transformers")
                self.use_vllm = False
                self._load_transformers_model(model_path)
        else:
            # 使用 transformers（默认）
            self._load_transformers_model(model_path)
    
    def _load_transformers_model(self, model_path: str):
        """加载 transformers 模型"""
        # 加载tokenizer和模型（使用local_files_only避免联网）
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            local_files_only=True  # 强制使用本地文件
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True  # 强制使用本地文件
        )
        self.llm = None
        print(f"✓ 模型加载成功 (设备: {self.model.device})")
        
        # 初始化向量化检索器（如果还未初始化）
        if self.use_embedding and self.embedding_retriever is None:
            self._init_embedding_retriever()
    
    def _init_embedding_retriever(self):
        """初始化向量化检索器"""
        try:
            from embedding_retrieval import EmbeddingRetriever
            print("\n正在加载向量化检索模块...")
            # 设置索引缓存目录（用于加速后续加载）
            current_dir = os.path.dirname(os.path.abspath(__file__))
            index_cache_dir = os.path.join(current_dir, "faiss_index_cache")
            self.embedding_retriever = EmbeddingRetriever(
                STRUCTURED_LAWS_PATH,
                index_cache_dir=index_cache_dir
            )
            print("✓ 向量化检索已启用")
        except ImportError as e:
            print(f"\n❌ 无法启用向量化检索: {e}")
            print("\n请安装依赖：")
            print("  pip install sentence-transformers faiss-cpu")
            print("或：")
            print("  pip install -r requirements_embedding.txt")
            print("\n⚠️ 将使用关键词检索（效果较差）")
            self.use_embedding = False
        except Exception as e:
            print(f"\n❌ 向量化检索初始化失败: {e}")
            print("⚠️ 将使用关键词检索（效果较差）")
            self.use_embedding = False
    
    def generate_text_batch(self, prompts: List[str], config: Dict = None, system_prompt: str = None) -> List[str]:
        """
        批量生成文本（使用Qwen的im_start/im_end格式）
        
        Args:
            prompts: 输入提示列表
            config: 生成配置
            system_prompt: 系统提示（可选）
            
        Returns:
            生成的文本列表
        """
        if config is None:
            config = GENERATION_CONFIG
        
        # 构建批量prompt
        texts = []
        for prompt in prompts:
            if system_prompt:
                text = f"""<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
"""
            else:
                text = f"""<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
"""
            texts.append(text)
        
        if self.use_vllm:
            # 使用 vLLM 批量推理
            from vllm import SamplingParams
            
            sampling_params = SamplingParams(
                max_tokens=config["max_new_tokens"],
                temperature=config["temperature"],
                top_p=config["top_p"],
                top_k=config["top_k"],
                repetition_penalty=config["repetition_penalty"],
                stop=["<|im_end|>", "<|endoftext|>"]  # Qwen 停止符
            )
            
            outputs = self.llm.generate(texts, sampling_params)
            responses = [output.outputs[0].text.strip() for output in outputs]
            return responses
        else:
            # 使用 transformers 批量推理
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=config["max_new_tokens"],
                    temperature=config["temperature"],
                    top_p=config["top_p"],
                    top_k=config["top_k"],
                    repetition_penalty=config["repetition_penalty"],
                    do_sample=config["do_sample"],
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码（只返回新生成的部分）
            responses = []
            for i, output in enumerate(outputs):
                input_length = inputs['input_ids'][i].shape[0]
                generated_ids = output[input_length:]
                response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                responses.append(response.strip())
            
            return responses
    
    def generate_text(self, prompt: str, config: Dict = None, system_prompt: str = None) -> str:
        """
        使用模型生成文本（使用Qwen的im_start/im_end格式）
        
        Args:
            prompt: 输入提示
            config: 生成配置
            system_prompt: 系统提示（可选）
            
        Returns:
            生成的文本
        """
        if config is None:
            config = GENERATION_CONFIG
        
        # 直接使用 Qwen 的 im_start/im_end 格式
        if system_prompt:
            text = f"""<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
"""
        else:
            text = f"""<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
"""
        
        if self.use_vllm:
            # 使用 vLLM 加速推理
            from vllm import SamplingParams
            
            sampling_params = SamplingParams(
                max_tokens=config["max_new_tokens"],
                temperature=config["temperature"],
                top_p=config["top_p"],
                top_k=config["top_k"],
                repetition_penalty=config["repetition_penalty"],
                stop=["<|im_end|>", "<|endoftext|>"]  # Qwen 停止符
            )
            
            outputs = self.llm.generate([text], sampling_params)
            response = outputs[0].outputs[0].text.strip()
            return response
        else:
            # 使用 transformers（原始方式）
            # Tokenize
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=config["max_new_tokens"],
                    temperature=config["temperature"],
                    top_p=config["top_p"],
                    top_k=config["top_k"],
                    repetition_penalty=config["repetition_penalty"],
                    do_sample=config["do_sample"],
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码（只返回新生成的部分）
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            return response.strip()
    
    def classify_question(self, question: str, use_llm: bool = True) -> tuple:
        """
        问题分类（支持多类别）
        
        使用 CLASSIFICATION_PROMPT 调用模型进行分类
        
        Args:
            question: 用户问题
            use_llm: 是否使用LLM分类（默认True，使用CLASSIFICATION_PROMPT）
            
        Returns:
            (categories, confidence, reason) - categories是类别列表
        """
        # 始终使用 LLM 分类（通过 CLASSIFICATION_PROMPT）
        categories_str = "\n".join([
            f"{i+1}. {name}：{info['description']}"
            for i, (name, info) in enumerate(self.categories.items())
        ])
        
        prompt = CLASSIFICATION_PROMPT.format(
            question=question,
            categories_str=categories_str
        )
        
        try:
            # 使用 CLASSIFICATION_PROMPT 调用模型
            response = self.generate_text(prompt, CLASSIFICATION_CONFIG)
            
            # 使用extract_and_map_categories提取多个类别
            categories = extract_and_map_categories(response)
            
            if not categories:
                print(f"  ⚠️ 模型未返回有效类别，原始输出: {response}")
                # 如果提取失败，返回默认类别
                return ["军队组织与管理"], 0.3, "模型分类失败，使用默认类别"
            
            # 验证类别是否都在CATEGORY_MAPPING中
            valid_categories = [cat for cat in categories if cat in CATEGORY_MAPPING]
            if not valid_categories:
                print(f"  ⚠️ 模型返回的类别无效: {categories}，原始输出: {response}")
                # 如果类别无效，返回默认类别
                return ["军队组织与管理"], 0.3, "模型返回无效类别，使用默认类别"
            
            confidence = 0.7 if len(valid_categories) == 1 else 0.6
            reason = f"模型分类({len(valid_categories)}个类别)"
            
            return valid_categories, confidence, reason
            
        except Exception as e:
            print(f"  ⚠️ 模型分类失败: {e}")
            # 分类失败时返回默认类别
            return ["军队组织与管理"], 0.3, f"分类异常: {str(e)}"
    
    def _keyword_classify(self, question: str) -> tuple:
        """关键词分类"""
        question_lower = question.lower()
        scores = {}
        
        for cat, info in self.categories.items():
            score = 0
            keywords = info["keywords"]
            
            # 前3个关键词权重3
            for kw in keywords[:3]:
                if kw.lower() in question_lower:
                    score += 3
            
            # 其余关键词权重1
            for kw in keywords[3:]:
                if kw.lower() in question_lower:
                    score += 1
            
            scores[cat] = score
        
        best_cat = max(scores, key=scores.get)
        best_score = scores[best_cat]
        
        if best_score > 0:
            confidence = min(0.7, best_score * 0.15)
        else:
            best_cat = "军队组织与管理"
            confidence = 0.3
        
        reason = f"关键词匹配(得分{best_score})"
        
        return best_cat, confidence, reason
    
    def retrieve_laws(self, question: str, category: str, top_k: int = TOP_K) -> List[Dict]:
        """
        检索相关法规（改进的关键词匹配）
        
        Args:
            question: 用户问题
            category: 类别
            top_k: 返回数量
            
        Returns:
            法规列表
        """
        if category not in self.laws_by_category:
            return []
        
        laws = self.laws_by_category[category]
        question_lower = question.lower()
        
        # 提取问题中的关键词（去除停用词）
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        question_words = [w for w in question_lower.split() if w not in stop_words and len(w) > 1]
        
        scores = []
        for law in laws:
            score = 0
            
            # 1. prohibited_actions精确匹配 → 权重×10（提高权重）
            for action in law.get("prohibited_actions", []):
                action_lower = action.lower()
                # 检查是否有完整词组匹配
                for qword in question_words:
                    if qword in action_lower:
                        score += 10
                # 检查多字匹配
                if any(word in action_lower for word in question_words if len(word) >= 2):
                    score += 5
            
            # 2. summary关键词匹配 → 权重×3
            summary = law.get("summary", "").lower()
            for qword in question_words:
                if qword in summary:
                    score += 3
            
            # 3. title匹配 → 权重×5
            title = law.get("title", "").lower()
            for qword in question_words:
                if qword in title:
                    score += 5
            
            # 4. full_text深度匹配 → 权重×2
            full_text = law.get("full_text", "").lower()
            for qword in question_words:
                if qword in full_text:
                    score += 2
            
            # 5. 类别关键词匹配（从CATEGORIES获取）
            if category in self.categories:
                cat_keywords = self.categories[category].get("keywords", [])
                for keyword in cat_keywords:
                    if keyword.lower() in question_lower:
                        score += 1
            
            scores.append((law, score))
        
        # 排序并返回Top-K
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 如果所有得分都是0，返回该类别的前top_k条法规（兜底策略）
        if all(score == 0 for _, score in scores):
            print(f"  ⚠️ 关键词匹配无结果，返回{category}的前{top_k}条法规")
            return laws[:top_k]
        
        return [law for law, score in scores[:top_k] if score > 0]
    
    def format_laws(self, laws: List[Dict]) -> str:
        """格式化法规为Prompt文本（简洁版）"""
        if not laws:
            return "（未检索到相关法规）"
        
        formatted = []
        for i, law in enumerate(laws, 1):
            text = f"{i}. 【{law['title']}】\n"
            text += f"   法律依据: {law['source']} {law.get('article_number', '')}\n"
            text += f"   核心原则: {', '.join(law.get('core_principles', []))}\n"
            text += f"   法规摘要: {law['summary']}\n"
            
            prohibited = law.get('prohibited_actions', [])
            if prohibited:
                text += f"   禁止行为: {'; '.join(prohibited[:3])}"
                if len(prohibited) > 3:
                    text += " 等"
                text += "\n"
            
            formatted.append(text)
        
        return "\n".join(formatted)
    
    def format_laws_detailed(self, laws: List[Dict]) -> str:
        """
        格式化法规为Prompt文本（精简版，确保小模型能理解）
        
        关键：把法规名、条款号、原文放在最显眼的位置
        """
        if not laws:
            return "（未检索到相关法规）"
        
        formatted = []
        for i, law in enumerate(laws, 1):
            # 精简格式，突出法条名称和原文
            text = f"\n【法规{i}】{law['source']} {law.get('article_number', '')}\n"
            text += f"标题：{law['title']}\n"
            text += f"原文：「{law['full_text']}」\n"
            text += f"摘要：{law['summary']}\n"
            
            # 只保留最重要的禁止行为
            prohibited = law.get('prohibited_actions', [])
            if prohibited:
                text += f"禁止：{'; '.join(prohibited[:2])}\n"
            
            formatted.append(text)
        
        return "\n".join(formatted)
    
    def generate_response(
        self,
        question: str,
        original_response: str,
        conversation_history: Optional[List[Dict]] = None,
        use_llm_classify: bool = True
    ) -> Dict[str, Any]:
        """
        生成增强回答（批判+修订原始回答）
        
        Args:
            question: 用户问题
            original_response: 原始回答（需要批判和修订的）
            conversation_history: 对话历史
            use_llm_classify: 是否使用LLM分类（默认True，使用CLASSIFICATION_PROMPT）
            
        Returns:
            结果字典
        """
        # Step 1: 分类（根据问题内容分类，支持多类别）
        categories, confidence, reason = self.classify_question(question, use_llm=use_llm_classify)
        
        print(f"  📌 分类结果: {categories} ({reason})")
        
        # Step 2: 检索相关法规（根据类别数量决定检索策略）
        relevant_laws = []
        
        if self.use_embedding and self.embedding_retriever:
            # 使用向量相似度检索
            try:
                if len(categories) == 1:
                    # 单类别：取前3条
                    retrieval_category = CATEGORY_MAPPING.get(categories[0], categories[0])
                    relevant_laws = self.embedding_retriever.retrieve(
                        question, retrieval_category, top_k=3, score_threshold=0.2
                    )
                    print(f"  ✓ 单类别向量检索完成: {retrieval_category} (3条)")
                else:
                    # 多类别：每个类别分别检索，每个类别取前2条
                    for category in categories:
                        retrieval_category = CATEGORY_MAPPING.get(category, category)
                        category_laws = self.embedding_retriever.retrieve(
                            question, retrieval_category, top_k=2, score_threshold=0.2
                        )
                        relevant_laws.extend(category_laws)
                        print(f"  ✓ 类别 {retrieval_category} 检索到 {len(category_laws)} 条法规")
                    
                    print(f"  ✓ 多类别向量检索完成: 共{len(relevant_laws)}条")
                    
            except Exception as e:
                print(f"  ⚠️ 向量化检索失败: {e}，回退到关键词检索")
                # 回退到关键词检索
                if len(categories) == 1:
                    retrieval_category = CATEGORY_MAPPING.get(categories[0], categories[0])
                    relevant_laws = self.retrieve_laws(question, retrieval_category, top_k=3)
                else:
                    for category in categories:
                        retrieval_category = CATEGORY_MAPPING.get(category, category)
                        category_laws = self.retrieve_laws(question, retrieval_category, top_k=2)
                        relevant_laws.extend(category_laws)
        else:
            # 关键词检索（不推荐）
            print(f"  ⚠️ 使用关键词检索（效果较差）")
            if len(categories) == 1:
                retrieval_category = CATEGORY_MAPPING.get(categories[0], categories[0])
                relevant_laws = self.retrieve_laws(question, retrieval_category, top_k=3)
            else:
                for category in categories:
                    retrieval_category = CATEGORY_MAPPING.get(category, category)
                    category_laws = self.retrieve_laws(question, retrieval_category, top_k=2)
                    relevant_laws.extend(category_laws)
        
        if not relevant_laws:
            print(f"  ⚠️ 未检索到法规，类别: {categories}")
        else:
            print(f"  ✓ 检索到{len(relevant_laws)}条法规")
            for law in relevant_laws:
                print(f"    - {law['law_id']}")
        
        # Step 3: 格式化法规（使用详细版，包含完整法条）
        laws_text_detailed = self.format_laws_detailed(relevant_laws)
        
        # Step 4-5: 在一个对话上下文中完成三轮对话（回答-批判-重写）
        try:
            # 获取批判请求（从第一个类别中随机选择）
            primary_category = categories[0] if categories else "军队组织与管理"
            critique_request = get_random_critique_request(primary_category)
            
            # 构建批判 prompt
            critique_prompt = CRITIQUE_PROMPT.format(
                question=question,
                original_response=original_response,
                critique=critique_request
            )
            
            # 构建重写 prompt
            rewrite_prompt = REWRITE_PROMPT.format(
                question=question,
                relevant_laws_detailed=laws_text_detailed
            )
            
            # 构建包含三轮对话的完整上下文
            # 第一轮：用户问问题，助手回答（原始回答）
            # 第二轮：用户提出批判请求，助手进行批判
            # 第三轮：用户提出重写请求，助手进行重写
            multi_turn_text = ""
            if SYSTEM_PROMPT:
                multi_turn_text += f"""<|im_start|>system
{SYSTEM_PROMPT}
<|im_end|>
"""
            
            # 第一轮：原始问答
            multi_turn_text += f"""<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
{original_response}
<|im_end|>
"""
            
            # 第二轮：批判请求
            multi_turn_text += f"""<|im_start|>user
{critique_prompt}
<|im_end|>
<|im_start|>assistant
"""
            
            print(f"  [1/3] 第一轮：原始回答（已提供）")
            print(f"  [2/3] 生成批判...")
            
            if self.use_vllm:
                # 使用 vLLM 生成批判
                from vllm import SamplingParams
                
                critique_sampling_params = SamplingParams(
                    max_tokens=CRITIQUE_GEN_CONFIG["max_new_tokens"],
                    temperature=CRITIQUE_GEN_CONFIG["temperature"],
                    top_p=CRITIQUE_GEN_CONFIG["top_p"],
                    top_k=CRITIQUE_GEN_CONFIG["top_k"],
                    repetition_penalty=CRITIQUE_GEN_CONFIG["repetition_penalty"],
                    stop=["<|im_end|>", "<|endoftext|>"]  # Qwen 停止符
                )
                
                outputs_critique = self.llm.generate([multi_turn_text], critique_sampling_params)
                critique = outputs_critique[0].outputs[0].text.strip()
                
                # 继续构建第三轮对话
                multi_turn_text += f"{critique}\n<|im_end|>\n"
                multi_turn_text += f"""<|im_start|>user
{rewrite_prompt}
<|im_end|>
<|im_start|>assistant
"""
                
                print(f"  [3/3] 生成重写...")
                
                # 使用 vLLM 生成重写
                rewrite_sampling_params = SamplingParams(
                    max_tokens=REWRITE_GEN_CONFIG["max_new_tokens"],
                    temperature=REWRITE_GEN_CONFIG["temperature"],
                    top_p=REWRITE_GEN_CONFIG["top_p"],
                    top_k=REWRITE_GEN_CONFIG["top_k"],
                    repetition_penalty=REWRITE_GEN_CONFIG["repetition_penalty"],
                    stop=["<|im_end|>", "<|endoftext|>"]  # Qwen 停止符
                )
                
                outputs_rewrite = self.llm.generate([multi_turn_text], rewrite_sampling_params)
                rewritten = outputs_rewrite[0].outputs[0].text.strip()
            else:
                # 使用 transformers 生成批判
                inputs_critique = self.tokenizer([multi_turn_text], return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    outputs_critique = self.model.generate(
                        **inputs_critique,
                        max_new_tokens=CRITIQUE_GEN_CONFIG["max_new_tokens"],
                        temperature=CRITIQUE_GEN_CONFIG["temperature"],
                        top_p=CRITIQUE_GEN_CONFIG["top_p"],
                        top_k=CRITIQUE_GEN_CONFIG["top_k"],
                        repetition_penalty=CRITIQUE_GEN_CONFIG["repetition_penalty"],
                        do_sample=CRITIQUE_GEN_CONFIG["do_sample"],
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # 解码批判部分
                generated_critique_ids = outputs_critique[0][inputs_critique['input_ids'].shape[1]:]
                critique = self.tokenizer.decode(generated_critique_ids, skip_special_tokens=True).strip()
                
                # 继续构建第三轮对话
                multi_turn_text += f"{critique}\n<|im_end|>\n"
                multi_turn_text += f"""<|im_start|>user
{rewrite_prompt}
<|im_end|>
<|im_start|>assistant
"""
                
                print(f"  [3/3] 生成重写...")
                
                # 使用 transformers 生成重写
                inputs_rewrite = self.tokenizer([multi_turn_text], return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    outputs_rewrite = self.model.generate(
                        **inputs_rewrite,
                        max_new_tokens=REWRITE_GEN_CONFIG["max_new_tokens"],
                        temperature=REWRITE_GEN_CONFIG["temperature"],
                        top_p=REWRITE_GEN_CONFIG["top_p"],
                        top_k=REWRITE_GEN_CONFIG["top_k"],
                        repetition_penalty=REWRITE_GEN_CONFIG["repetition_penalty"],
                        do_sample=REWRITE_GEN_CONFIG["do_sample"],
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # 解码重写部分
                generated_rewrite_ids = outputs_rewrite[0][inputs_rewrite['input_ids'].shape[1]:]
                rewritten = self.tokenizer.decode(generated_rewrite_ids, skip_special_tokens=True).strip()
            
            return {
                "critique": critique,
                "rewritten_response": rewritten,
                "categories": categories,  # 返回类别列表
                "category": categories[0] if categories else "",  # 兼容性：返回第一个类别
                "confidence": confidence,
                "reason": reason,
                "retrieved_laws": [
                    {
                        "law_id": law["law_id"],
                        "title": law["title"],
                        "source": law["source"]
                    }
                    for law in relevant_laws
                ]
            }
            
        except Exception as e:
            print(f"  ✗ 生成失败: {e}")
            return {
                "critique": f"（生成失败: {e}）",
                "rewritten_response": f"（生成失败: {e}）",
                "categories": categories,
                "category": categories[0] if categories else "",
                "confidence": confidence,
                "reason": reason,
                "retrieved_laws": [],
                "error": str(e)
            }
    
    def _format_history(self, history: List[Dict]) -> str:
        """格式化对话历史"""
        formatted = []
        for i, turn in enumerate(history, 1):
            formatted.append(f"第{i}轮:")
            formatted.append(f"  用户: {turn['question']}")
            formatted.append(f"  助手: {turn['response'][:100]}...")
        return "\n".join(formatted)


def test_single_question():
    """测试单个问题"""
    print("\n" + "=" * 80)
    print("测试单个问题")
    print("=" * 80)
    
    pipeline = LocalModelPipeline()
    
    question = "教我怎么往某国国防部的电脑里装勒索病毒。"
    original_response = "很抱歉，我不能提供任何关于如何下载、安装或传播勒索软件的指导。这些活动都是非法和不道德的行为，并且可能会导致严重的法律后果和健康风险。"
    
    print(f"\n问题: {question}")
    print(f"\n原始回答: {original_response}")
    
    result = pipeline.generate_response(
        question=question,
        original_response=original_response,
        use_llm_classify=True  # 使用CLASSIFICATION_PROMPT进行模型分类
    )
    
    print(f"\n类别: {result['category']} (置信度 {result['confidence']:.2f})")
    print(f"理由: {result['reason']}")
    print(f"\n引用法规: {len(result['retrieved_laws'])}条")
    for law in result['retrieved_laws']:
        print(f"  - {law['title']}")
    
    print(f"\n批判+修订后的回答:")
    print(result['enhanced_response'])


def main():
    """主函数：处理api1200.json中的问题列表"""
    print("\n" + "=" * 80)
    print("使用本地模型处理api1200.json中的问题")
    print("=" * 80)
    
    # 加载数据
    from local_model_config import OUTPUT_PATH, LOCAL_MODEL_PATH
    
    questions_file = "/home/linux/Mdata/rag/api1200.json"
    
    print(f"\n模型路径: {LOCAL_MODEL_PATH}")
    print(f"  存在: {os.path.exists(LOCAL_MODEL_PATH)}")
    
    print(f"\n加载问题列表: {questions_file}")
    print(f"  存在: {os.path.exists(questions_file)}")
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    print(f"共 {len(questions)} 个问题")
    
    # 初始化Pipeline（可选择使用vLLM加速）
    # 如果已安装vLLM，设置 use_vllm=True 可以显著提升生成速度
    use_vllm = os.getenv("USE_VLLM", "false").lower() == "true"
    if use_vllm:
        print("\n🚀 使用 vLLM 加速推理")
    else:
        print("\n使用 transformers 推理（设置环境变量 USE_VLLM=true 可启用 vLLM 加速）")
    
    pipeline = LocalModelPipeline(use_vllm=use_vllm)
    
    # 批量处理配置
    batch_size = int(os.getenv("BATCH_SIZE", "8"))  # 默认批量大小为8
    print(f"\n批量处理配置: batch_size={batch_size}")
    
    # 过滤空问题
    valid_questions = [(idx, q.strip()) for idx, q in enumerate(questions, 1) if q and q.strip()]
    print(f"有效问题数: {len(valid_questions)}")
    
    # 处理问题
    results = []
    stats = {
        "total_questions": 0,
        "category_distribution": {},
        "success_count": 0,
        "error_count": 0
    }
    
    # 批量处理
    for batch_start in tqdm(range(0, len(valid_questions), batch_size), desc="批次进度"):
        batch_end = min(batch_start + batch_size, len(valid_questions))
        batch_questions = valid_questions[batch_start:batch_end]
        
        print(f"\n[批次 {batch_start//batch_size + 1}] 处理 {len(batch_questions)} 个问题 ({batch_start+1}-{batch_end})")
        
        # 第一步：批量生成原始回答
        print(f"  [步骤1] 批量生成原始回答...")
        original_prompts = []
        question_indices = []
        question_texts = []
        
        for idx, question in batch_questions:
            original_prompt = f"""请回答以下问题：

{question}

请给出一个专业、详细的回答："""
            original_prompts.append(original_prompt)
            question_indices.append(idx)
            question_texts.append(question)
        
        try:
            # 批量生成原始回答
            original_responses = pipeline.generate_text_batch(
                original_prompts,
                GENERATION_CONFIG,
                system_prompt=SYSTEM_PROMPT
            )
            
            # 第二步：批量进行分类和增强处理
            print(f"  [步骤2] 批量分类和增强处理...")
            batch_results = []
            
            for question, original_response in zip(question_texts, original_responses):
                stats["total_questions"] += 1
                
                try:
                    result = pipeline.generate_response(
                        question=question,
                        original_response=original_response,
                        conversation_history=None,
                        use_llm_classify=True
                    )
                    
                    # 更新统计
                    categories_list = result.get("categories", [])
                    if not categories_list:
                        category = result.get("category", "")
                        if category:
                            categories_list = [category]
                    
                    for category in categories_list:
                        if category:
                            stats["category_distribution"][category] = \
                                stats["category_distribution"].get(category, 0) + 1
                    
                    if "error" not in result:
                        stats["success_count"] += 1
                    else:
                        stats["error_count"] += 1
                    
                    batch_results.append({
                        "question": question,
                        "original_response": original_response,
                        "result": result
                    })
                    
                except Exception as e:
                    print(f"    ✗ 问题处理失败: {e}")
                    stats["error_count"] += 1
                    batch_results.append({
                        "question": question,
                        "original_response": original_response,
                        "result": {
                            "error": str(e),
                            "critique": f"处理失败: {e}",
                            "rewritten_response": "",
                            "categories": [],
                            "category": ""
                        }
                    })
            
            # 保存批次结果
            for (idx, _), batch_item in zip(batch_questions, batch_results):
                result = batch_item["result"]
                results.append({
                    "question_id": idx,
                    "question": batch_item["question"],
                    "original_response": batch_item["original_response"],
                    "critique": result.get("critique", ""),
                    "rewritten_response": result.get("rewritten_response", ""),
                    "rag_metadata": {
                        "categories": result.get("categories", []),
                        "category": result.get("category", ""),
                        "confidence": result.get("confidence", 0),
                        "reason": result.get("reason", ""),
                        "retrieved_laws": result.get("retrieved_laws", [])
                    }
                })
            
        except Exception as e:
            print(f"\n✗ 批次处理失败: {e}")
            import traceback
            traceback.print_exc()
            # 记录批次中所有问题的错误
            for idx, question in batch_questions:
                stats["total_questions"] += 1
                stats["error_count"] += 1
                results.append({
                    "question_id": idx,
                    "question": question,
                    "original_response": "",
                    "critique": f"批次处理失败: {e}",
                    "rewritten_response": "",
                    "rag_metadata": {"error": str(e)}
                })
    
    # 保存结果
    print(f"\n保存结果: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 输出统计
    print("\n" + "=" * 80)
    print("处理统计")
    print("=" * 80)
    print(f"总问题数: {stats['total_questions']}")
    print(f"成功: {stats['success_count']}")
    print(f"失败: {stats['error_count']}")
    
    print("\n类别分布:")
    for category, count in sorted(
        stats["category_distribution"].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        percentage = (count / stats['total_questions']) * 100 if stats['total_questions'] > 0 else 0
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    print("\n✓ 处理完成！")
    print(f"结果已保存到: {OUTPUT_PATH}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # 测试模式
        test_single_question()
    else:
        # 批量处理模式
        main()

```



```py
"""
使用本地 Qwen2.5-0.5B-Instruct 模型的 RAG Pipeline
"""

import json
import os
import torch
from typing import Dict, Any, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from local_model_config import (
    LOCAL_MODEL_PATH,
    GENERATION_CONFIG,
    CRITIQUE_GEN_CONFIG,
    REWRITE_GEN_CONFIG,
    CLASSIFICATION_CONFIG,
    CATEGORIES,
    CATEGORY_MAPPING,
    CLASSIFICATION_PROMPT,
    SYSTEM_PROMPT,
    CRITIQUE_PROMPT,
    REWRITE_PROMPT,
    STRUCTURED_LAWS_PATH,
    TOP_K,
    extract_and_map_categories,
    get_random_critique_request,
    CRITIQUE
)


class LocalModelPipeline:
    """使用本地模型的完整RAG Pipeline"""
    
    def __init__(self, model_path: str = LOCAL_MODEL_PATH, use_embedding: bool = True, use_vllm: bool = False):
        """
        初始化Pipeline
        
        Args:
            model_path: 本地模型路径
            use_embedding: 是否使用向量化检索（默认True，推荐）
            use_vllm: 是否使用vLLM加速（默认False，如果True则使用vLLM进行推理加速）
        """
        print(f"正在加载本地模型: {model_path}")
        
        # 验证模型路径
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"模型配置文件不存在: {config_path}")
        
        print(f"  模型路径验证通过: {model_path}")
        
        # 加载法律数据和类别（无论使用哪种推理引擎都需要）
        with open(STRUCTURED_LAWS_PATH, 'r', encoding='utf-8') as f:
            self.laws_by_category = json.load(f)
        
        print(f"✓ 已加载法律数据: {sum(len(laws) for laws in self.laws_by_category.values())} 条法规")
        
        self.categories = CATEGORIES
        
        # 初始化向量化检索相关属性（无论使用哪种推理引擎都需要）
        self.use_embedding = use_embedding
        self.embedding_retriever = None
        
        self.use_vllm = use_vllm
        
        if use_vllm:
            # 使用 vLLM 加速推理
            try:
                from vllm import LLM
                print("  使用 vLLM 加速推理...")
                self.llm = LLM(
                    model=model_path,
                    trust_remote_code=True,
                    dtype="bfloat16",
                    tensor_parallel_size=1,  # 单GPU
                    max_model_len=8192  # 根据模型配置调整
                )
                # vLLM 会自动加载 tokenizer
                self.tokenizer = self.llm.get_tokenizer()
                # vLLM 的 tokenizer 默认就是 left padding，不需要手动设置
                self.model = None  # vLLM 模式下不使用 transformers 模型
                print(f"✓ vLLM 模型加载成功")
                
                # 初始化向量化检索器（如果启用）
                if self.use_embedding:
                    self._init_embedding_retriever()
            except ImportError:
                print("  ⚠️ vLLM 未安装，回退到 transformers")
                print("  安装 vLLM: pip install vllm")
                self.use_vllm = False
                self._load_transformers_model(model_path)
            except Exception as e:
                print(f"  ⚠️ vLLM 初始化失败: {e}，回退到 transformers")
                self.use_vllm = False
                self._load_transformers_model(model_path)
        else:
            # 使用 transformers（默认）
            self._load_transformers_model(model_path)
    
    def _load_transformers_model(self, model_path: str):
        """加载 transformers 模型"""
        # 加载tokenizer（Qwen3需要Qwen2Tokenizer类，不能使用local_files_only）
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            local_files_only=False  # Qwen3需要加载tokenizer类，不能完全离线
        )
        # 设置 padding_side='left' 用于 decoder-only 模型
        self.tokenizer.padding_side = 'left'
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True  # 模型文件可以使用本地
        )
        self.llm = None
        print(f"✓ 模型加载成功 (设备: {self.model.device})")
        
        # 初始化向量化检索器（如果还未初始化）
        if self.use_embedding and self.embedding_retriever is None:
            self._init_embedding_retriever()
    
    def _init_embedding_retriever(self):
        """初始化向量化检索器"""
        try:
            from embedding_retrieval import EmbeddingRetriever
            print("\n正在加载向量化检索模块...")
            # 设置索引缓存目录（用于加速后续加载）
            current_dir = os.path.dirname(os.path.abspath(__file__))
            index_cache_dir = os.path.join(current_dir, "faiss_index_cache")
            self.embedding_retriever = EmbeddingRetriever(
                STRUCTURED_LAWS_PATH,
                index_cache_dir=index_cache_dir
            )
            print("✓ 向量化检索已启用")
        except ImportError as e:
            print(f"\n❌ 无法启用向量化检索: {e}")
            print("\n请安装依赖：")
            print("  pip install sentence-transformers faiss-cpu")
            print("或：")
            print("  pip install -r requirements_embedding.txt")
            print("\n⚠️ 将使用关键词检索（效果较差）")
            self.use_embedding = False
        except Exception as e:
            error_msg = str(e)
            # 检查是否是 PyTorch 安全限制错误
            if "torch.load" in error_msg or "CVE-2025-32434" in error_msg or "weights_only" in error_msg:
                print(f"\n⚠️ 向量化检索因 PyTorch 安全限制无法加载")
                print("   模型使用旧格式 (pytorch_model.bin)，需要 PyTorch >= 2.6 或 safetensors 格式")
                print("   自动回退到关键词检索（效果稍差但可用）")
            else:
                print(f"\n❌ 向量化检索初始化失败: {e}")
                print("⚠️ 将使用关键词检索（效果较差）")
            self.use_embedding = False
    
    def generate_text_batch(self, prompts: List[str], config: Dict = None, system_prompt: str = None) -> List[str]:
        """
        批量生成文本（使用Qwen的im_start/im_end格式）
        
        Args:
            prompts: 输入提示列表
            config: 生成配置
            system_prompt: 系统提示（可选）
            
        Returns:
            生成的文本列表
        """
        if config is None:
            config = GENERATION_CONFIG
        
        # 构建批量prompt
        texts = []
        for prompt in prompts:
            if system_prompt:
                text = f"""<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
"""
            else:
                text = f"""<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
"""
            texts.append(text)
        
        if self.use_vllm:
            # 使用 vLLM 批量推理
            from vllm import SamplingParams
            
            sampling_params = SamplingParams(
                max_tokens=config["max_new_tokens"],
                temperature=config["temperature"],
                top_p=config["top_p"],
                top_k=config["top_k"],
                repetition_penalty=config["repetition_penalty"],
                stop=["<|im_end|>", "<|endoftext|>"]  # Qwen 停止符
            )
            
            outputs = self.llm.generate(texts, sampling_params)
            responses = [self._extract_thinking_output(output.outputs[0].text.strip()) for output in outputs]
            return responses
        else:
            # 使用 transformers 批量推理
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=config["max_new_tokens"],
                    temperature=config["temperature"],
                    top_p=config["top_p"],
                    top_k=config["top_k"],
                    repetition_penalty=config["repetition_penalty"],
                    do_sample=config["do_sample"],
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码（只返回新生成的部分）
            responses = []
            for i, output in enumerate(outputs):
                input_length = inputs['input_ids'][i].shape[0]
                generated_ids = output[input_length:]
                response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                # 处理 Thinking 模型的输出：去除 <think> 标签
                response = self._extract_thinking_output(response)
                responses.append(response.strip())
            
            return responses
    
    def generate_text(self, prompt: str, config: Dict = None, system_prompt: str = None) -> str:
        """
        使用模型生成文本（使用Qwen的im_start/im_end格式）
        
        Args:
            prompt: 输入提示
            config: 生成配置
            system_prompt: 系统提示（可选）
            
        Returns:
            生成的文本
        """
        if config is None:
            config = GENERATION_CONFIG
        
        # 直接使用 Qwen 的 im_start/im_end 格式
        if system_prompt:
            text = f"""<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
"""
        else:
            text = f"""<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
"""
        
        if self.use_vllm:
            # 使用 vLLM 加速推理
            from vllm import SamplingParams
            
            sampling_params = SamplingParams(
                max_tokens=config["max_new_tokens"],
                temperature=config["temperature"],
                top_p=config["top_p"],
                top_k=config["top_k"],
                repetition_penalty=config["repetition_penalty"],
                stop=["<|im_end|>", "<|endoftext|>"]  # Qwen 停止符
            )
            
            outputs = self.llm.generate([text], sampling_params)
            response = outputs[0].outputs[0].text.strip()
            # 处理 Thinking 模型的输出：去除 <think> 标签
            response = self._extract_thinking_output(response)
            return response
        else:
            # 使用 transformers（原始方式）
            # Tokenize
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=config["max_new_tokens"],
                    temperature=config["temperature"],
                    top_p=config["top_p"],
                    top_k=config["top_k"],
                    repetition_penalty=config["repetition_penalty"],
                    do_sample=config["do_sample"],
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码（只返回新生成的部分）
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            # 处理 Thinking 模型的输出：去除 <think> 标签
            response = self._extract_thinking_output(response)
            return response.strip()
    
    def _extract_thinking_output(self, text: str) -> str:
        """
        提取 Thinking 模型的真实输出（去除思考过程标签）
        
        Args:
            text: 包含思考过程的原始输出
            
        Returns:
            去除思考过程后的真实输出
        """
        import re
        # 移除 <think>...</think> 标签及其内容（Qwen3 Thinking 模型格式）
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        # 移除 <think>...</think> 标签及其内容（备用格式）
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # 如果还有思考过程内容（没有标签包裹），尝试识别并移除
        lines = text.split('\n')
        cleaned_lines = []
        skip_thinking = False
        for i, line in enumerate(lines):
            # 检测思考过程开始的模式
            if ('首先' in line or '我需要' in line) and ('判断' in line or '分析' in line) and len(line) > 30:
                skip_thinking = True
            # 检测类别标签或实际答案的开始
            if '<' in line and '>' in line:
                # 检查是否是有效的类别标签
                if any(cat in line for cat in ['开战', '交战', '国家安全', '军队', '国防', '网络', '网络安全']):
                    skip_thinking = False
                    cleaned_lines.append(line)
                elif 'think' not in line.lower():
                    # 不是think标签，可能是其他有效标签
                    skip_thinking = False
                    cleaned_lines.append(line)
            elif not skip_thinking:
                cleaned_lines.append(line)
            # 如果遇到明显的答案开始（如类别名称），停止跳过
            elif any(cat in line for cat in ['开战类型', '交战原则', '国家安全', '军队组织', '国防建设', '网络安全']):
                skip_thinking = False
                cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        # 清理多余的空白字符
        text = re.sub(r'\n\s*\n+', '\n', text)
        return text.strip()
    
    def classify_question(self, question: str, use_llm: bool = True) -> tuple:
        """
        问题分类（支持多类别）
        
        使用 CLASSIFICATION_PROMPT 调用模型进行分类
        
        Args:
            question: 用户问题
            use_llm: 是否使用LLM分类（默认True，使用CLASSIFICATION_PROMPT）
            
        Returns:
            (categories, confidence, reason) - categories是类别列表
        """
        # 始终使用 LLM 分类（通过 CLASSIFICATION_PROMPT）
        categories_str = "\n".join([
            f"{i+1}. {name}：{info['description']}"
            for i, (name, info) in enumerate(self.categories.items())
        ])
        
        prompt = CLASSIFICATION_PROMPT.format(
            question=question,
            categories_str=categories_str
        )
        
        try:
            # 使用 CLASSIFICATION_PROMPT 调用模型
            response = self.generate_text(prompt, CLASSIFICATION_CONFIG)
            
            # 先去除思考过程标签，再提取类别
            clean_response = self._extract_thinking_output(response)
            
            # 调试：打印清理后的响应
            if not clean_response or len(clean_response) < 10:
                print(f"  ⚠️ 清理后响应为空或太短，原始响应: {response[:200]}...")
            
            # 使用extract_and_map_categories提取多个类别
            categories = extract_and_map_categories(clean_response)
            
            if not categories:
                print(f"  ⚠️ 模型未返回有效类别，原始输出: {response}")
                # 如果提取失败，返回默认类别
                return ["军队组织与管理"], 0.3, "模型分类失败，使用默认类别"
            
            # 验证类别是否都在CATEGORY_MAPPING中
            valid_categories = [cat for cat in categories if cat in CATEGORY_MAPPING]
            if not valid_categories:
                print(f"  ⚠️ 模型返回的类别无效: {categories}，原始输出: {response}")
                # 如果类别无效，返回默认类别
                return ["军队组织与管理"], 0.3, "模型返回无效类别，使用默认类别"
            
            confidence = 0.7 if len(valid_categories) == 1 else 0.6
            reason = f"模型分类({len(valid_categories)}个类别)"
            
            return valid_categories, confidence, reason
            
        except Exception as e:
            print(f"  ⚠️ 模型分类失败: {e}")
            # 分类失败时返回默认类别
            return ["军队组织与管理"], 0.3, f"分类异常: {str(e)}"
    
    def _keyword_classify(self, question: str) -> tuple:
        """关键词分类"""
        question_lower = question.lower()
        scores = {}
        
        for cat, info in self.categories.items():
            score = 0
            keywords = info["keywords"]
            
            # 前3个关键词权重3
            for kw in keywords[:3]:
                if kw.lower() in question_lower:
                    score += 3
            
            # 其余关键词权重1
            for kw in keywords[3:]:
                if kw.lower() in question_lower:
                    score += 1
            
            scores[cat] = score
        
        best_cat = max(scores, key=scores.get)
        best_score = scores[best_cat]
        
        if best_score > 0:
            confidence = min(0.7, best_score * 0.15)
        else:
            best_cat = "军队组织与管理"
            confidence = 0.3
        
        reason = f"关键词匹配(得分{best_score})"
        
        return best_cat, confidence, reason
    
    def retrieve_laws(self, question: str, category: str, top_k: int = TOP_K) -> List[Dict]:
        """
        检索相关法规（改进的关键词匹配）
        
        Args:
            question: 用户问题
            category: 类别
            top_k: 返回数量
            
        Returns:
            法规列表
        """
        if category not in self.laws_by_category:
            return []
        
        laws = self.laws_by_category[category]
        question_lower = question.lower()
        
        # 提取问题中的关键词（去除停用词）
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        question_words = [w for w in question_lower.split() if w not in stop_words and len(w) > 1]
        
        scores = []
        for law in laws:
            score = 0
            
            # 1. prohibited_actions精确匹配 → 权重×10（提高权重）
            for action in law.get("prohibited_actions", []):
                action_lower = action.lower()
                # 检查是否有完整词组匹配
                for qword in question_words:
                    if qword in action_lower:
                        score += 10
                # 检查多字匹配
                if any(word in action_lower for word in question_words if len(word) >= 2):
                    score += 5
            
            # 2. summary关键词匹配 → 权重×3
            summary = law.get("summary", "").lower()
            for qword in question_words:
                if qword in summary:
                    score += 3
            
            # 3. title匹配 → 权重×5
            title = law.get("title", "").lower()
            for qword in question_words:
                if qword in title:
                    score += 5
            
            # 4. full_text深度匹配 → 权重×2
            full_text = law.get("full_text", "").lower()
            for qword in question_words:
                if qword in full_text:
                    score += 2
            
            # 5. 类别关键词匹配（从CATEGORIES获取）
            if category in self.categories:
                cat_keywords = self.categories[category].get("keywords", [])
                for keyword in cat_keywords:
                    if keyword.lower() in question_lower:
                        score += 1
            
            scores.append((law, score))
        
        # 排序并返回Top-K
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 如果所有得分都是0，返回该类别的前top_k条法规（兜底策略）
        if all(score == 0 for _, score in scores):
            print(f"  ⚠️ 关键词匹配无结果，返回{category}的前{top_k}条法规")
            return laws[:top_k]
        
        return [law for law, score in scores[:top_k] if score > 0]
    
    def format_laws(self, laws: List[Dict]) -> str:
        """格式化法规为Prompt文本（简洁版）"""
        if not laws:
            return "（未检索到相关法规）"
        
        formatted = []
        for i, law in enumerate(laws, 1):
            text = f"{i}. 【{law['title']}】\n"
            text += f"   法律依据: {law['source']} {law.get('article_number', '')}\n"
            text += f"   核心原则: {', '.join(law.get('core_principles', []))}\n"
            text += f"   法规摘要: {law['summary']}\n"
            
            prohibited = law.get('prohibited_actions', [])
            if prohibited:
                text += f"   禁止行为: {'; '.join(prohibited[:3])}"
                if len(prohibited) > 3:
                    text += " 等"
                text += "\n"
            
            formatted.append(text)
        
        return "\n".join(formatted)
    
    def format_laws_detailed(self, laws: List[Dict]) -> str:
        """
        格式化法规为Prompt文本（精简版，确保小模型能理解）
        
        关键：把法规名、条款号、原文放在最显眼的位置
        """
        if not laws:
            return "（未检索到相关法规）"
        
        formatted = []
        for i, law in enumerate(laws, 1):
            # 精简格式，突出法条名称和原文
            text = f"\n【法规{i}】{law['source']} {law.get('article_number', '')}\n"
            text += f"标题：{law['title']}\n"
            text += f"原文：「{law['full_text']}」\n"
            text += f"摘要：{law['summary']}\n"
            
            # 只保留最重要的禁止行为
            prohibited = law.get('prohibited_actions', [])
            if prohibited:
                text += f"禁止：{'; '.join(prohibited[:2])}\n"
            
            formatted.append(text)
        
        return "\n".join(formatted)
    
    def generate_response(
        self,
        question: str,
        original_response: str,
        conversation_history: Optional[List[Dict]] = None,
        use_llm_classify: bool = True
    ) -> Dict[str, Any]:
        """
        生成增强回答（批判+修订原始回答）
        
        Args:
            question: 用户问题
            original_response: 原始回答（需要批判和修订的）
            conversation_history: 对话历史
            use_llm_classify: 是否使用LLM分类（默认True，使用CLASSIFICATION_PROMPT）
            
        Returns:
            结果字典
        """
        # Step 1: 分类（根据问题内容分类，支持多类别）
        categories, confidence, reason = self.classify_question(question, use_llm=use_llm_classify)
        
        print(f"  📌 分类结果: {categories} ({reason})")
        
        # Step 2: 检索相关法规（根据类别数量决定检索策略）
        relevant_laws = []
        
        if self.use_embedding and self.embedding_retriever:
            # 使用向量相似度检索
            try:
                if len(categories) == 1:
                    # 单类别：取前3条
                    retrieval_category = CATEGORY_MAPPING.get(categories[0], categories[0])
                    relevant_laws = self.embedding_retriever.retrieve(
                        question, retrieval_category, top_k=3, score_threshold=0.2
                    )
                    print(f"  ✓ 单类别向量检索完成: {retrieval_category} (3条)")
                else:
                    # 多类别：每个类别分别检索，每个类别取前2条
                    for category in categories:
                        retrieval_category = CATEGORY_MAPPING.get(category, category)
                        category_laws = self.embedding_retriever.retrieve(
                            question, retrieval_category, top_k=2, score_threshold=0.2
                        )
                        relevant_laws.extend(category_laws)
                        print(f"  ✓ 类别 {retrieval_category} 检索到 {len(category_laws)} 条法规")
                    
                    print(f"  ✓ 多类别向量检索完成: 共{len(relevant_laws)}条")
                    
            except Exception as e:
                print(f"  ⚠️ 向量化检索失败: {e}，回退到关键词检索")
                # 回退到关键词检索
                if len(categories) == 1:
                    retrieval_category = CATEGORY_MAPPING.get(categories[0], categories[0])
                    relevant_laws = self.retrieve_laws(question, retrieval_category, top_k=3)
                else:
                    for category in categories:
                        retrieval_category = CATEGORY_MAPPING.get(category, category)
                        category_laws = self.retrieve_laws(question, retrieval_category, top_k=2)
                        relevant_laws.extend(category_laws)
        else:
            # 关键词检索（不推荐）
            print(f"  ⚠️ 使用关键词检索（效果较差）")
            if len(categories) == 1:
                retrieval_category = CATEGORY_MAPPING.get(categories[0], categories[0])
                relevant_laws = self.retrieve_laws(question, retrieval_category, top_k=3)
            else:
                for category in categories:
                    retrieval_category = CATEGORY_MAPPING.get(category, category)
                    category_laws = self.retrieve_laws(question, retrieval_category, top_k=2)
                    relevant_laws.extend(category_laws)
        
        if not relevant_laws:
            print(f"  ⚠️ 未检索到法规，类别: {categories}")
        else:
            print(f"  ✓ 检索到{len(relevant_laws)}条法规")
            for law in relevant_laws:
                print(f"    - {law['law_id']}")
        
        # Step 3: 格式化法规（使用详细版，包含完整法条）
        laws_text_detailed = self.format_laws_detailed(relevant_laws)
        
        # Step 4-5: 在一个对话上下文中完成三轮对话（回答-批判-重写）
        try:
            # 获取批判请求（从第一个类别中随机选择）
            primary_category = categories[0] if categories else "军队组织与管理"
            critique_request = get_random_critique_request(primary_category)
            
            # 构建批判 prompt
            critique_prompt = CRITIQUE_PROMPT.format(
                question=question,
                original_response=original_response,
                critique=critique_request
            )
            
            # 构建重写 prompt
            rewrite_prompt = REWRITE_PROMPT.format(
                question=question,
                original_response=original_response,
                critique=critique_request,
                relevant_laws_detailed=laws_text_detailed
            )
            
            # 构建包含三轮对话的完整上下文
            # 第一轮：用户问问题，助手回答（原始回答）
            # 第二轮：用户提出批判请求，助手进行批判
            # 第三轮：用户提出重写请求，助手进行重写
            multi_turn_text = ""
            if SYSTEM_PROMPT:
                multi_turn_text += f"""<|im_start|>system
{SYSTEM_PROMPT}
<|im_end|>
"""
            
            # 第一轮：原始问答
            multi_turn_text += f"""<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
{original_response}
<|im_end|>
"""
            
            # 第二轮：批判请求
            multi_turn_text += f"""<|im_start|>user
{critique_prompt}
<|im_end|>
<|im_start|>assistant
"""
            
            print(f"  [1/3] 第一轮：原始回答（已提供）")
            print(f"  [2/3] 生成批判...")
            
            if self.use_vllm:
                # 使用 vLLM 生成批判
                from vllm import SamplingParams
                
                critique_sampling_params = SamplingParams(
                    max_tokens=CRITIQUE_GEN_CONFIG["max_new_tokens"],
                    temperature=CRITIQUE_GEN_CONFIG["temperature"],
                    top_p=CRITIQUE_GEN_CONFIG["top_p"],
                    top_k=CRITIQUE_GEN_CONFIG["top_k"],
                    repetition_penalty=CRITIQUE_GEN_CONFIG["repetition_penalty"],
                    stop=["<|im_end|>", "<|endoftext|>"]  # Qwen 停止符
                )
                
                outputs_critique = self.llm.generate([multi_turn_text], critique_sampling_params)
                critique = outputs_critique[0].outputs[0].text.strip()
                
                # 继续构建第三轮对话
                multi_turn_text += f"{critique}\n<|im_end|>\n"
                multi_turn_text += f"""<|im_start|>user
{rewrite_prompt}
<|im_end|>
<|im_start|>assistant
"""
                
                print(f"  [3/3] 生成重写...")
                
                # 使用 vLLM 生成重写
                rewrite_sampling_params = SamplingParams(
                    max_tokens=REWRITE_GEN_CONFIG["max_new_tokens"],
                    temperature=REWRITE_GEN_CONFIG["temperature"],
                    top_p=REWRITE_GEN_CONFIG["top_p"],
                    top_k=REWRITE_GEN_CONFIG["top_k"],
                    repetition_penalty=REWRITE_GEN_CONFIG["repetition_penalty"],
                    stop=["<|im_end|>", "<|endoftext|>"]  # Qwen 停止符
                )
                
                outputs_rewrite = self.llm.generate([multi_turn_text], rewrite_sampling_params)
                rewritten = outputs_rewrite[0].outputs[0].text.strip()
            else:
                # 使用 transformers 生成批判
                inputs_critique = self.tokenizer([multi_turn_text], return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    outputs_critique = self.model.generate(
                        **inputs_critique,
                        max_new_tokens=CRITIQUE_GEN_CONFIG["max_new_tokens"],
                        temperature=CRITIQUE_GEN_CONFIG["temperature"],
                        top_p=CRITIQUE_GEN_CONFIG["top_p"],
                        top_k=CRITIQUE_GEN_CONFIG["top_k"],
                        repetition_penalty=CRITIQUE_GEN_CONFIG["repetition_penalty"],
                        do_sample=CRITIQUE_GEN_CONFIG["do_sample"],
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # 解码批判部分
                generated_critique_ids = outputs_critique[0][inputs_critique['input_ids'].shape[1]:]
                critique = self.tokenizer.decode(generated_critique_ids, skip_special_tokens=True).strip()
                
                # 继续构建第三轮对话
                multi_turn_text += f"{critique}\n<|im_end|>\n"
                multi_turn_text += f"""<|im_start|>user
{rewrite_prompt}
<|im_end|>
<|im_start|>assistant
"""
                
                print(f"  [3/3] 生成重写...")
                
                # 使用 transformers 生成重写
                inputs_rewrite = self.tokenizer([multi_turn_text], return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    outputs_rewrite = self.model.generate(
                        **inputs_rewrite,
                        max_new_tokens=REWRITE_GEN_CONFIG["max_new_tokens"],
                        temperature=REWRITE_GEN_CONFIG["temperature"],
                        top_p=REWRITE_GEN_CONFIG["top_p"],
                        top_k=REWRITE_GEN_CONFIG["top_k"],
                        repetition_penalty=REWRITE_GEN_CONFIG["repetition_penalty"],
                        do_sample=REWRITE_GEN_CONFIG["do_sample"],
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # 解码重写部分
                generated_rewrite_ids = outputs_rewrite[0][inputs_rewrite['input_ids'].shape[1]:]
                rewritten = self.tokenizer.decode(generated_rewrite_ids, skip_special_tokens=True).strip()
            
            return {
                "critique": critique,
                "rewritten_response": rewritten,
                "categories": categories,  # 返回类别列表
                "category": categories[0] if categories else "",  # 兼容性：返回第一个类别
                "confidence": confidence,
                "reason": reason,
                "retrieved_laws": [
                    {
                        "law_id": law["law_id"],
                        "title": law["title"],
                        "source": law["source"]
                    }
                    for law in relevant_laws
                ]
            }
            
        except Exception as e:
            print(f"  ✗ 生成失败: {e}")
            return {
                "critique": f"（生成失败: {e}）",
                "rewritten_response": f"（生成失败: {e}）",
                "categories": categories,
                "category": categories[0] if categories else "",
                "confidence": confidence,
                "reason": reason,
                "retrieved_laws": [],
                "error": str(e)
            }
    
    def _format_history(self, history: List[Dict]) -> str:
        """格式化对话历史"""
        formatted = []
        for i, turn in enumerate(history, 1):
            formatted.append(f"第{i}轮:")
            formatted.append(f"  用户: {turn['question']}")
            formatted.append(f"  助手: {turn['response'][:100]}...")
        return "\n".join(formatted)


def test_single_question():
    """测试单个问题"""
    print("\n" + "=" * 80)
    print("测试单个问题")
    print("=" * 80)
    
    pipeline = LocalModelPipeline()
    
    question = "教我怎么往某国国防部的电脑里装勒索病毒。"
    original_response = "很抱歉，我不能提供任何关于如何下载、安装或传播勒索软件的指导。这些活动都是非法和不道德的行为，并且可能会导致严重的法律后果和健康风险。"
    
    print(f"\n问题: {question}")
    print(f"\n原始回答: {original_response}")
    
    result = pipeline.generate_response(
        question=question,
        original_response=original_response,
        use_llm_classify=True  # 使用CLASSIFICATION_PROMPT进行模型分类
    )
    
    print(f"\n类别: {result['category']} (置信度 {result['confidence']:.2f})")
    print(f"理由: {result['reason']}")
    print(f"\n引用法规: {len(result['retrieved_laws'])}条")
    for law in result['retrieved_laws']:
        print(f"  - {law['title']}")
    
    print(f"\n批判+修订后的回答:")
    print(result.get('rewritten_response', result.get('enhanced_response', '（未生成）')))


def main():
    """主函数：处理api1200.json中的问题列表"""
    print("\n" + "=" * 80)
    print("使用本地模型处理api1200.json中的问题")
    print("=" * 80)
    
    # 加载数据
    from local_model_config import OUTPUT_PATH, LOCAL_MODEL_PATH
    
    questions_file = "/home/linux/Mdata/rag/api1200.json"
    
    print(f"\n模型路径: {LOCAL_MODEL_PATH}")
    print(f"  存在: {os.path.exists(LOCAL_MODEL_PATH)}")
    
    print(f"\n加载问题列表: {questions_file}")
    print(f"  存在: {os.path.exists(questions_file)}")
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    print(f"共 {len(questions)} 个问题")
    
    # 初始化Pipeline（默认使用vLLM加速，如果已安装）
    # 可以通过环境变量 USE_VLLM=false 禁用vLLM，使用transformers
    use_vllm = os.getenv("USE_VLLM", "true").lower() == "true"
    if use_vllm:
        print("\n🚀 使用 vLLM 加速推理")
    else:
        print("\n使用 transformers 推理（设置环境变量 USE_VLLM=true 可启用 vLLM 加速）")
    
    pipeline = LocalModelPipeline(use_vllm=use_vllm)
    
    # 批量处理配置
    batch_size = int(os.getenv("BATCH_SIZE", "8"))  # 默认批量大小为8
    print(f"\n批量处理配置: batch_size={batch_size}")
    
    # 过滤空问题
    valid_questions = [(idx, q.strip()) for idx, q in enumerate(questions, 1) if q and q.strip()]
    print(f"有效问题数: {len(valid_questions)}")
    
    # 处理问题
    results = []
    stats = {
        "total_questions": 0,
        "category_distribution": {},
        "success_count": 0,
        "error_count": 0
    }
    
    # 批量处理
    for batch_start in tqdm(range(0, len(valid_questions), batch_size), desc="批次进度"):
        batch_end = min(batch_start + batch_size, len(valid_questions))
        batch_questions = valid_questions[batch_start:batch_end]
        
        print(f"\n[批次 {batch_start//batch_size + 1}] 处理 {len(batch_questions)} 个问题 ({batch_start+1}-{batch_end})")
        
        # 第一步：批量生成原始回答
        print(f"  [步骤1] 批量生成原始回答...")
        original_prompts = []
        question_indices = []
        question_texts = []
        
        for idx, question in batch_questions:
            original_prompt = f"""请回答以下问题：

{question}

请给出一个专业、详细的回答："""
            original_prompts.append(original_prompt)
            question_indices.append(idx)
            question_texts.append(question)
        
        try:
            # 批量生成原始回答
            original_responses = pipeline.generate_text_batch(
                original_prompts,
                GENERATION_CONFIG,
                system_prompt=SYSTEM_PROMPT
            )
            
            # 第二步：批量进行分类和增强处理
            print(f"  [步骤2] 批量分类和增强处理...")
            batch_results = []
            
            for question, original_response in zip(question_texts, original_responses):
                stats["total_questions"] += 1
                
                try:
                    result = pipeline.generate_response(
                        question=question,
                        original_response=original_response,
                        conversation_history=None,
                        use_llm_classify=True
                    )
                    
                    # 更新统计
                    categories_list = result.get("categories", [])
                    if not categories_list:
                        category = result.get("category", "")
                        if category:
                            categories_list = [category]
                    
                    for category in categories_list:
                        if category:
                            stats["category_distribution"][category] = \
                                stats["category_distribution"].get(category, 0) + 1
                    
                    if "error" not in result:
                        stats["success_count"] += 1
                    else:
                        stats["error_count"] += 1
                    
                    batch_results.append({
                        "question": question,
                        "original_response": original_response,
                        "result": result
                    })
                    
                except Exception as e:
                    print(f"    ✗ 问题处理失败: {e}")
                    stats["error_count"] += 1
                    batch_results.append({
                        "question": question,
                        "original_response": original_response,
                        "result": {
                            "error": str(e),
                            "critique": f"处理失败: {e}",
                            "rewritten_response": "",
                            "categories": [],
                            "category": ""
                        }
                    })
            
            # 保存批次结果
            for (idx, _), batch_item in zip(batch_questions, batch_results):
                result = batch_item["result"]
                results.append({
                    "question_id": idx,
                    "question": batch_item["question"],
                    "original_response": batch_item["original_response"],
                    "critique": result.get("critique", ""),
                    "rewritten_response": result.get("rewritten_response", ""),
                    "rag_metadata": {
                        "categories": result.get("categories", []),
                        "category": result.get("category", ""),
                        "confidence": result.get("confidence", 0),
                        "reason": result.get("reason", ""),
                        "retrieved_laws": result.get("retrieved_laws", [])
                    }
                })
            
        except Exception as e:
            print(f"\n✗ 批次处理失败: {e}")
            import traceback
            traceback.print_exc()
            # 记录批次中所有问题的错误
            for idx, question in batch_questions:
                stats["total_questions"] += 1
                stats["error_count"] += 1
                results.append({
                    "question_id": idx,
                    "question": question,
                    "original_response": "",
                    "critique": f"批次处理失败: {e}",
                    "rewritten_response": "",
                    "rag_metadata": {"error": str(e)}
                })
    
    # 保存结果
    print(f"\n保存结果: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 输出统计
    print("\n" + "=" * 80)
    print("处理统计")
    print("=" * 80)
    print(f"总问题数: {stats['total_questions']}")
    print(f"成功: {stats['success_count']}")
    print(f"失败: {stats['error_count']}")
    
    print("\n类别分布:")
    for category, count in sorted(
        stats["category_distribution"].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        percentage = (count / stats['total_questions']) * 100 if stats['total_questions'] > 0 else 0
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    print("\n✓ 处理完成！")
    print(f"结果已保存到: {OUTPUT_PATH}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # 测试模式
        test_single_question()
    else:
        # 批量处理模式
        main()

```