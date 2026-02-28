# Law Text2Vec 模型使用指南

## 模型信息

- **模型名称**: ChatLaw-Text2Vec
- **Hugging Face ID**: `chestnutlzj/ChatLaw-Text2Vec`
- **模型链接**: https://huggingface.co/chestnutlzj/ChatLaw-Text2Vec
- **用途**: 法律相关文本的相似度计算，可用于制作向量数据库

## 方法一：直接从 Hugging Face 使用（推荐）

最简单的方式是直接使用 Hugging Face 模型 ID，SentenceTransformer 会自动下载：

```bash
cd /home/linux/Mdata/rag
python test_law_text2vec.py --model_path chestnutlzj/ChatLaw-Text2Vec
```

如果网络较慢或需要使用镜像站点：

```bash
# 设置 Hugging Face 镜像站点（可选）
export HF_ENDPOINT=https://hf-mirror.com

# 运行测试
python test_law_text2vec.py --model_path chestnutlzj/ChatLaw-Text2Vec
```

## 方法二：先下载到本地再使用

如果需要离线使用或避免重复下载：

### 1. 下载模型

```bash
cd /home/linux/Mdata/rag

# 使用默认保存路径
python download_law_text2vec.py

# 或指定自定义保存路径
python download_law_text2vec.py --save_dir ./models/ChatLaw-Text2Vec
```

### 2. 使用本地模型运行测试

```bash
python test_law_text2vec.py --model_path ./models/ChatLaw-Text2Vec
```

## 完整参数示例

```bash
python test_law_text2vec.py \
    --model_path chestnutlzj/ChatLaw-Text2Vec \
    --questions_path api1200.json \
    --laws_path structured_laws_by_category.json \
    --output_path law_text2vec_results.json \
    --top_k 5 \
    --batch_size 8 \
    --device cuda
```

## 参数说明

- `--model_path`: 模型路径或 Hugging Face 模型 ID（必需）
- `--questions_path`: 问题文件路径（默认: `turn1_questions.json`）
- `--laws_path`: 法条文件路径（默认: `structured_laws_by_category.json`）
- `--output_path`: 输出文件路径（默认: `law_text2vec_results.json`）
- `--top_k`: 每个问题返回的法条数量（默认: 5）
- `--batch_size`: 批处理大小（默认: 32，可根据 GPU 内存调整）
- `--device`: 设备类型，`cuda` 或 `cpu`（默认: 自动检测）

## 输出格式

结果会保存为 JSON 文件，格式如下：

```json
[
  {
    "question": "问题文本",
    "question_id": 0,
    "top_laws": [
      {
        "law_id": "法条ID",
        "title": "法条标题",
        "category": "法条类别",
        "full_text": "法条全文",
        "summary": "法条摘要",
        "source": "法条来源",
        "article_number": "条款号",
        "similarity_score": 0.95,
        "rank": 1
      },
      ...
    ]
  },
  ...
]
```

## 注意事项

1. **网络连接**: 如果直接从 Hugging Face 加载，需要稳定的网络连接
2. **镜像站点**: 如果下载较慢，可以使用 Hugging Face 镜像站点：
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```
3. **GPU 内存**: 根据 GPU 内存调整 `--batch_size` 参数
4. **依赖安装**: 确保已安装所需依赖：
   ```bash
   pip install sentence-transformers torch tqdm
   ```

## 示例输出

运行后会显示：
- 模型加载进度
- 问题编码进度
- 法条编码进度
- 相似度计算进度
- 前 3 个问题的结果示例

## 类内法条 RAG（问题 + law.json）

使用项目内问题与 `law/law.json` 做**类内**法条相似度检索（每个问题只在该问题的 categories 对应类别下检索法条），输出供 `rewrite_with_laws.py` 使用的相似度文件：

```bash
cd /home/linux/Mdata/rag

# 使用默认路径：问题=apimakequestion/.../local_assessed_20260218_005211.json，法条=law/law.json
python question_law_rag.py

# 指定路径与 top_k
python question_law_rag.py \
  --questions /home/linux/Mdata/apimakequestion/generate_output/0218new/local_assessed_20260218_005211.json \
  --laws /home/linux/Mdata/law/law.json \
  --output /home/linux/Mdata/apimakequestion/generate_output/0218new/question_law_similarity_0218new.json \
  --top_k 5
```

- 输出格式：`[{ "id": 0, "question": "...", "top_laws": [{ "law_name", "article_name", "content", "similarity_score", "rank" }] }, ...]`
- 重写时：`rewrite_with_laws.py --input <assessed.json> --similarity <上面输出的 json>`；若输入条目不包含 `id`，会按索引与相似度文件对齐；若只有 `safety_answer_api`，会自动当作 `safety_answer` 使用。

## 参考

- [Hugging Face 模型页面](https://huggingface.co/chestnutlzj/ChatLaw-Text2Vec)
- [ChatLaw 论文](https://arxiv.org/abs/2306.16092)
