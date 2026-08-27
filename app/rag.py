"""RAG knowledge base: document loading, vector store, and retrieval."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    BSHTMLLoader,
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
)
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from modelscope import snapshot_download

# 混合检索依赖（可选）：没装 jieba/rank_bm25 时自动降级为纯向量检索
try:
    import jieba
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    jieba = None
    BM25Okapi = None
    _BM25_AVAILABLE = False

from app.config import Settings


SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf", ".docx", ".csv", ".xlsx", ".html", ".htm"}
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
COLLECTION_NAME = "agent_knowledge"


@dataclass(frozen=True)
class RagPaths:
    """All filesystem paths the RAG system needs, derived from Settings."""

    knowledge_dir: Path
    vector_db_dir: Path
    manifest_path: Path
    local_embedding_dir: Path


def build_rag_paths(settings: Settings) -> RagPaths:
    """Derive RAG paths from the application Settings."""
    knowledge_dir = settings.workspace_dir / "knowledge"
    vector_db_dir = settings.workspace_dir / "chroma_db"
    return RagPaths(
        knowledge_dir=knowledge_dir,
        vector_db_dir=vector_db_dir,
        manifest_path=vector_db_dir / "knowledge_manifest.json",
        local_embedding_dir=settings.workspace_dir / "models" / "bge-small-zh-v1.5",
    )


def load_documents(file_path: Path) -> list[Document]:
    """Load a single file into Document objects, choosing a loader by suffix."""
    suffix = file_path.suffix.lower()

    if suffix in {".md", ".txt"}:
        text = file_path.read_text(encoding="utf-8").strip()
        if not text:
            return []
        return [Document(page_content=text, metadata={"source": file_path.name})]

    if suffix == ".pdf":
        documents = PyPDFLoader(str(file_path), mode="page").load()
        for document in documents:
            document.metadata["source"] = file_path.name
            document.metadata["page"] = document.metadata.get("page", 0) + 1
        return [d for d in documents if d.page_content.strip()]

    if suffix == ".docx":
        documents = Docx2txtLoader(str(file_path)).load()
        for document in documents:
            document.metadata["source"] = file_path.name
        return [d for d in documents if d.page_content.strip()]

    if suffix == ".csv":
        documents = CSVLoader(str(file_path), autodetect_encoding=True).load()
        for document in documents:
            document.metadata["source"] = file_path.name
            row = document.metadata.get("row")
            if isinstance(row, int):
                document.metadata["row"] = row + 2
        return [d for d in documents if d.page_content.strip()]

    if suffix == ".xlsx":
        sheets = pd.read_excel(file_path, sheet_name=None, dtype=str, keep_default_na=False)
        documents = []
        for sheet_name, dataframe in sheets.items():
            for excel_row, (_, row) in enumerate(dataframe.iterrows(), start=2):
                row_text = "\n".join(
                    f"{column}: {str(value).strip()}"
                    for column, value in row.items()
                    if str(value).strip()
                )
                if row_text:
                    documents.append(
                        Document(
                            page_content=row_text,
                            metadata={
                                "source": file_path.name,
                                "sheet_name": sheet_name,
                                "row": excel_row,
                            },
                        )
                    )
        return documents

    if suffix in {".html", ".htm"}:
        documents = BSHTMLLoader(str(file_path)).load()
        for document in documents:
            document.metadata["source"] = file_path.name
        return [d for d in documents if d.page_content.strip()]

    raise ValueError(f"暂不支持的知识库文件格式：{file_path.suffix}")


def _chunk_fingerprint(document: Document) -> str:
    """内容指纹：用（来源+正文）唯一标识一个文本块，用于跨检索路去重。"""
    source = document.metadata.get("source", "")
    return f"{source}|{document.page_content}"


class KnowledgeBase:
    """Thread-safe wrapper around the Chroma vector store.

    Manages the lazy build / incremental rebuild / cached load lifecycle
    of the knowledge base, using a manifest of file hashes to detect
    when documents changed.
    """

    def __init__(self, paths: RagPaths):
        self._paths = paths
        self._vector_store = None
        self._lock = threading.Lock()
        self._chunks: list[Document] = []   # 文本块（BM25 关键词检索要用）
        self._bm25 = None                   # BM25Okapi 索引

    def _get_embeddings(self):
        model_weight = self._paths.local_embedding_dir / "model.safetensors"
        if model_weight.is_file():
            print(f"使用已下载的本地 embedding 模型：{self._paths.local_embedding_dir}")
            model_dir = str(self._paths.local_embedding_dir)
        else:
            print(f"正在通过 ModelScope 下载 embedding 模型：{EMBEDDING_MODEL}")
            model_dir = snapshot_download(
                model_id=EMBEDDING_MODEL,
                local_dir=str(self._paths.local_embedding_dir),
                allow_patterns=["*.json", "model.safetensors", "vocab.txt"],
            )
        return HuggingFaceEmbeddings(
            model_name=model_dir,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def ensure_ready(self, force_rebuild: bool = False):
        """Make the vector store available, rebuilding it if documents changed."""
        with self._lock:
            return self._ensure_ready_locked(force_rebuild)

    def _ensure_ready_locked(self, force_rebuild: bool):
        knowledge_dir = self._paths.knowledge_dir
        if not knowledge_dir.is_dir():
            raise RuntimeError(f"知识库目录不存在：{knowledge_dir}")

        knowledge_files = sorted(
            file_path
            for file_path in knowledge_dir.iterdir()
            if file_path.is_file()
            and file_path.suffix.lower() in SUPPORTED_SUFFIXES
        )
        if not knowledge_files:
            raise RuntimeError(f"知识库目录中没有支持的文件：{knowledge_dir}")

        documents: list[Document] = []
        current_manifest: dict[str, str] = {}
        for file_path in knowledge_files:
            file_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
            current_manifest[file_path.name] = file_hash
            documents.extend(load_documents(file_path))
        if not documents:
            raise RuntimeError("知识库文档都是空的。")

        old_manifest = None
        if self._paths.manifest_path.is_file():
            try:
                old_manifest = json.loads(self._paths.manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                print("知识库指纹记录损坏，将完整重建。")

        needs_rebuild = (
            force_rebuild
            or not self._paths.vector_db_dir.is_dir()
            or old_manifest != current_manifest
        )

        if not needs_rebuild and self._vector_store is not None:
            return self._vector_store

        embeddings = self._get_embeddings()

        if needs_rebuild:
            self._vector_store = None
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
            chunks = splitter.split_documents(documents)
            self._chunks = chunks          # 保存文本块，供 BM25 关键词检索用
            self._build_bm25(chunks)       # 构建关键词索引

            if self._paths.vector_db_dir.exists():
                old_store = Chroma(
                    collection_name=COLLECTION_NAME,
                    embedding_function=embeddings,
                    persist_directory=str(self._paths.vector_db_dir),
                )
                old_store.delete_collection()

            vector_store = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=str(self._paths.vector_db_dir),
            )
            vector_store.add_documents(chunks)

            self._paths.manifest_path.write_text(
                json.dumps(current_manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._vector_store = vector_store
            print(
                f"知识库已完整重建：{len(documents)} 个文档，{len(chunks)} 个文本块。"
            )
            return self._vector_store

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(self._paths.vector_db_dir),
            create_collection_if_not_exists=False,
        )
        if not vector_store.get(limit=1)["ids"]:
            raise RuntimeError("持久化知识库为空，将在下次调用时重新构建。")
        self._vector_store = vector_store
        # 持久化加载时，用源文档重新切分并构建 BM25 索引（只做文本，不做 embedding）
        if _BM25_AVAILABLE and not self._chunks:
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
            self._chunks = splitter.split_documents(documents)
            self._build_bm25(self._chunks)
        print("知识库文件未变化，已加载持久化知识库。")
        return self._vector_store

    def _build_bm25(self, chunks: list[Document]) -> None:
        """用 jieba 分词构建 BM25 关键词索引。"""
        if not _BM25_AVAILABLE:
            self._bm25 = None
            return
        tokenized = [list(jieba.cut(chunk.page_content)) for chunk in chunks]
        self._bm25 = BM25Okapi(tokenized)

    def _bm25_search(self, query: str, k: int) -> list[Document]:
        """BM25 关键词路：精确匹配查询词，返回 top-k 文本块。"""
        if self._bm25 is None or not self._chunks:
            return []
        tokens = list(jieba.cut(query))
        scores = self._bm25.get_scores(tokens)
        top_indices = scores.argsort()[::-1][:k]
        return [self._chunks[i] for i in top_indices if scores[i] > 0]

    def search(self, query: str, k: int = 3, score_threshold: float = 0.2) -> str:
        """混合检索：向量语义路 + BM25 关键词路，RRF 融合排序后格式化返回。

        - 向量路：找"语义相近但用词不同"的片段（如"保修多久" → 保修政策）
        - BM25 路：找"关键词精确命中"的片段（如型号 XL-2026、专有名词）
        - RRF（Reciprocal Rank Fusion）：两路结果按排名打分合并，双路都命中的排更前
        """
        try:
            with self._lock:
                vector_store = self._ensure_ready_locked(False)
                # 1. 向量路：多召回一些（k*2），显式过滤低于阈值的
                vector_results = [
                    (d, s)
                    for d, s in vector_store.similarity_search_with_relevance_scores(
                        query, k=k * 2, score_threshold=score_threshold
                    )
                    if s >= score_threshold
                ]
                # 2. BM25 关键词路
                bm25_hits = self._bm25_search(query, k=k * 2)

            # 3. RRF 融合：两路结果按"排名"打分（第0名=1/60，第1名=1/61...）
            rrf_scores: dict[str, float] = {}
            doc_map: dict[str, Document] = {}
            for rank, (document, _score) in enumerate(vector_results):
                key = _chunk_fingerprint(document)
                rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (60 + rank)
                doc_map[key] = document
            for rank, document in enumerate(bm25_hits):
                key = _chunk_fingerprint(document)
                rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (60 + rank)
                doc_map[key] = document

            ranked = sorted(rrf_scores.items(), key=lambda item: -item[1])[:k]
            if not ranked:
                return "知识库中没有检索到足够相关的内容，不能依据知识库回答。"

            items = []
            for index, (key, rrf_score) in enumerate(ranked, start=1):
                document = doc_map[key]
                source = document.metadata.get("source", "未知来源")
                # 位置精度：把加载时存好的 page / sheet_name / row 显示出来
                location = ""
                page = document.metadata.get("page")
                sheet = document.metadata.get("sheet_name")
                row = document.metadata.get("row")
                if page:
                    location += f"第{page}页"
                if sheet:
                    location += f"「{sheet}」表第{row}行"
                location_suffix = f"｜位置：{location}" if location else ""
                items.append(
                    f"【片段 {index}｜综合相关度：{rrf_score * 100:.2f}｜来源：{source}{location_suffix}】\n"
                    f"{document.page_content}"
                )
            return "\n\n".join(items)

        except Exception as error:
            return f"知识库检索失败：{type(error).__name__} - {error}"
