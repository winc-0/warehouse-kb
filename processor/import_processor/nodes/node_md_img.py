# processor/import_processor/nodes/node_md_img.py
import base64
import json
import logging
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Tuple, List, Deque, Dict

from minio import Minio
from minio.deleteobjects import DeleteObject

from config.minio_config import minio_config
from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, FileProcessingError
from processor.import_processor.state import ImportGraphState
from utils.llm_utils import build_client
from utils.minio_utils import get_minio_client


def _env_int(name: str, default: int) -> int:
    """读整数环境变量，非法值回落到默认。"""
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_flag(name: str, default: bool = True) -> bool:
    """读布尔环境变量：1/true/on/yes 为真，0/false/off/no 为假。"""
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "on", "yes", "y")


class NodeMDImg(BaseNode):
    """
    MarkDown图片处理节点：多模态图片理解
    """

    name = "node_md_img"

    def process(self, state: ImportGraphState):

        """
        MD文件图片处理核心节点
        核心流程：
        1. 获取MD内容、文件路径、图片文件夹路径
        2. 扫描图片文件夹，筛选MD中实际引用的支持格式图片
        3. 调用多模态大模型为图片生成内容摘要
        4. 将图片上传至MinIO，替换MD中本地图片路径为MinIO访问URL，并填充图片摘要
        5. 备份原MD文件，保存处理后的新MD文件并更新状态

        :param state: md_path、md_content
        :return: md_path、md_content
        """

        # 步骤1：初始化数据，获取MD核心信息
        md_content, md_path_obj, images_dir = self._step_1_get_content(state)

        # 先把原文写回 state。
        # 直接喂 .md 时（没有走 node_pdf_to_md），state 里本来没有 md_content，
        # 而下面几条分支都是提前 return——一旦命中「无图片文件夹」，
        # md_content 就永远是空，下游 node_document_split 直接报「文件内容不能为空」。
        # 这里兜底写入，保证无论走哪条分支，下游拿到的都是真实正文。
        state["md_content"] = md_content

        # 0. 总开关：MinIO 不可用（比如时钟漂移导致 S3 签名失效）时整段跳过。
        #    此时既不调 VL 也不上传，直接把正文里的图片引用去掉，保持文本干净。
        if not _env_flag("MD_IMG_UPLOAD", True):
            stripped = self._strip_image_refs(md_content)
            if stripped != md_content:
                self.logger.info("MD_IMG_UPLOAD=off，跳过图片处理并移除正文图片引用")
                state["md_content"] = stripped
            return state

        if not images_dir.exists():
            self.logger.info("无图片文件夹，跳过图片处理")
            return state

        # 步骤2：扫描并筛选MD中引用的图片
        target_images = self._step_2_scan_images(md_content, images_dir)
        if not target_images:
            self.logger.info("未检测到MD中引用了图片，跳过图片处理")
            return state

        # 步骤3：调用多模态大模型生成图片摘要
        # MD_IMG_SUMMARY=off 可整段跳过（批量灌语料时省时间，图片仍会上传并保留 URL）
        if _env_flag("MD_IMG_SUMMARY", True):
            summaries = self._step_3_generate_summaries(md_path_obj.stem, target_images)
        else:
            self.logger.info("MD_IMG_SUMMARY=off，跳过图片摘要，仅上传图片")
            summaries = {img_file: "" for img_file, _, _ in target_images}

        # 步骤4：上传图片至MinIO，替换MD图片路径并填充摘要
        new_md_content = self._step_4_upload_and_replace(md_path_obj.stem, target_images, summaries, md_content)

        # 步骤5：备份并保存新MD文件
        new_md_file_name = self._step_5_backup_new_md_file(state['md_path'], new_md_content)

        # 步骤6：更新state状态值
        state["md_content"] = new_md_content
        state["md_path"] = new_md_file_name

        return state

    def _strip_image_refs(self, md_content: str) -> str:
        """
        把 Markdown 里的图片引用整段删掉。

        用于 `MD_IMG_UPLOAD=off`：图片不传到 MinIO，留着 `![](images/xxx.jpg)`
        这种本地路径在正文里只是噪音，进 embedding 会稀释语义，不如去掉。
        """
        # ![alt](path) 或 ![alt](path) 带 link 包裹的嵌套形式，先处理最简单的一层
        cleaned = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md_content)
        # 压缩因删除产生的连续空行
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned

    def _step_5_backup_new_md_file(self, origin_md_path: str, md_content: str) -> str:
        """
        步骤5：将处理后的MD内容保存为新文件（原文件不变，避免数据丢失）
        新文件命名规则：原文件名 + _new.md（如test.md → test_new.md）
        :param origin_md_path: 原始MD文件完整路径
        :param md_content: 处理后的新MD内容
        :return: 新MD文件的完整路径
        """

        new_md_file_name = os.path.splitext(origin_md_path)[0] + "_new.md"
        with open(new_md_file_name, "w", encoding="utf-8") as f:
            f.write(md_content)

        return new_md_file_name

    def _step_4_upload_and_replace(
            self,
            doc_stem: str,
            target_images: List[Tuple[str, str, Tuple[str, str]]],
            summaries: Dict[str, str],
            md_content: str
    ) -> str:
        """
        步骤 4: 上传图片并合并信息，然后替换 Markdown 中的内容。

        流程：
        1. 确定 MinIO 上的上传目录（按文档名隔离）。
        2. 清理该目录下的旧数据。
        3. 批量上传图片。
        4. 合并“图片摘要”和“图片URL”。
        5. 替换 Markdown 文本中的图片引用。
        :param doc_stem: 文档文件名（不含后缀），作为MinIO上传子目录名（按文档隔离）
        :param target_images: 待处理图片列表，元素为(图片文件名, 图片完整路径, 图片上下文)
        :param summaries: 图片摘要字典，键：图片文件名，值：内容摘要
        :param md_content: 原始MD文件内容
        :return: 图片引用替换后的新MD内容
        """

        # 1. 获取MinIO客户端
        minio_client = get_minio_client()
        if minio_client is None:
            # MinIO 不可用时不要把整篇文档搞挂：跳过上传，正文保留原始图片引用
            self.logger.error("MinIO 不可用，跳过图片上传（正文保留原图片引用，不影响文本入库）")
            return md_content

        # 2. 获取图片上传路径
        minio_dir = minio_config.img_dir
        upload_dir = f"{minio_dir}/{doc_stem}".replace(" ", "")

        # 3. 清理已有目录
        self._clean_minio_dir(minio_client, upload_dir)

        # 4.批量上传
        urls = self._upload_images_batch(minio_client, upload_dir, target_images)

        # 5. 合并图片摘要和URL，过滤上传失败的图片
        image_info = self._merge_summary_and_url(summaries, urls)

        # 6. 替换MD中的图片引用
        md_content = self._process_md_file(md_content, image_info)

        return md_content

    def _process_md_file(self, md_content: str, image_info: Dict[str, Tuple[str, str]]) -> str:
        """
        核心功能：替换MD内容中的本地图片引用为MinIO远程引用
        替换规则：![原描述](本地路径) → ![图片摘要](MinIO访问URL)
        :param md_content: 原始MD文件内容
        :param image_info: 合并后的图片信息字典，键：图片文件名，值：(摘要, URL)
        :return: 替换后的新MD内容
        """

        for image_file, (summary, new_url) in image_info.items():
            pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)")
            #sub(替换规则, 将要被替换的内容
            md_content = pattern.sub(lambda m: f"![{summary}]({new_url})", md_content)

        return md_content

    def _merge_summary_and_url(self, summaries: Dict[str, str], urls: Dict[str, str]) -> Dict[str, Tuple[str, str]]:
        """
        合并图片摘要字典和URL字典，过滤掉上传失败无URL的图片
        :param summaries: 图片摘要字典，键：图片文件名，值：内容摘要
        :param urls: 图片URL字典，键：图片文件名，值：MinIO访问URL
        :return: 合并后的图片信息字典，键：图片文件名，值：(摘要, URL)元组
        """
        image_info = {}
        for image_file, summary in summaries.items():
            if url := urls.get(image_file):
                image_info[image_file] = (summary, url)
        return image_info

    def _upload_images_batch(self, minio_client: Minio, upload_dir: str, target_images: List[Tuple]) -> Dict[str, str]:
        """
        批量上传待处理图片至MinIO，返回图片文件名与访问URL的映射关系
        :param minio_client: 初始化完成的MinIO客户端对象
        :param upload_dir: MinIO上传根目录
        :param target_images: 待处理图片列表，元素为(图片文件名, 图片完整路径, 图片上下文)
        :return: 图片URL字典，键：图片文件名，值：MinIO访问URL
        """
        urls = {}
        for img_file, img_path, _ in target_images:
            object_name = f"{upload_dir}/{img_file}"
            urls[img_file] = self._upload_to_minio(minio_client, img_path, object_name)
        return urls

    def _upload_to_minio(self, minio_client: Minio, local_path: str, object_name: str) -> str | None:
        """
        将单张本地图片上传至MinIO对象存储，并返回公网可访问URL
        :param minio_client: 初始化完成的MinIO客户端对象
        :param local_path: 图片本地完整路径
        :param object_name: MinIO中要存储的对象名称
        :return: 图片MinIO访问URL（上传失败返回None）
        """

        try:
            content_type = os.path.splitext(local_path)[1][1:]
            minio_client.fput_object(
                bucket_name = minio_config.bucket_name,
                object_name=object_name,
                file_path=local_path,
                content_type=f"image/{content_type}",
            )

            #组织图片url
            # 注意：object_name 可能带前导斜杠（由 minio_config.img_dir 的前导 '/' 传递而来），
            # 直接拼接会产生 //knowledge-base//upload-images 这类双斜杠。MinIO 存储会自动
            # 规范化掉前导斜杠，但 URL 字符串会残留双斜杠。这里统一 lstrip 保证 URL 干净。
            _obj = object_name.lstrip("/")
            url = f"http://{minio_config.endpoint}/{minio_config.bucket_name}/{_obj}"
            return url
        except Exception as e:
            # 一定要把异常本身打出来，否则批量导入时只知道「失败了」却查不出原因
            self.logger.error(f"上传图片失败：{local_path}，错误：{type(e).__name__}: {e}")

    def _clean_minio_dir(self, minio_client: Minio, update_dir: str) -> None:

        try:
            # 1. 获取将要被删除的图片列表
            objects_to_delete = minio_client.list_objects(minio_config.bucket_name, update_dir, recursive=True)

            delete_list = [DeleteObject(obj.object_name) for obj in objects_to_delete]
            errors = minio_client.remove_objects(
                minio_config.bucket_name,
                #需要一个 [DeleteObject]
                delete_list,
            )

            #2. 打印错误信息
            for error in errors:
                self.logger.error(f"删除图片错误：{error}")

        except Exception as e:
            self.logger.error(f"MinIO连接失败，错误原因：{e}")

    def _step_3_generate_summaries(self, doc_stem: str, target_images: List[Tuple[str, str, Tuple[str, str]]]) -> Dict[str, str]:
        """
        步骤3：批量为待处理图片生成内容摘要，带API速率限制防止触发大模型限流
        :param doc_stem: 文档文件名（不含后缀），作为大模型prompt上下文
        :param targets: 待处理图片列表，元素为(图片文件名, 图片完整路径, 图片上下文)
        :param requests_per_minute: 每分钟最大API请求数，默认9次（按大模型限制调整）
        :return: 图片摘要字典，键：图片文件名，值：图片内容摘要
        """

        # 1. 定义摘要
        summaries = {}

        # 2. 定义双端队列
        request_deque = deque()

        # 3. 客户端只建一次，别在循环里反复初始化
        chat_model = build_client(role="vl")

        # 4. 循环处理图片
        for img_file, img_path, context in target_images:

            # 4.1 限速（VL_RPM 可调；默认 60，原来硬编码 10 太保守，
            #     一份手册上百张图时会卡在 60 秒等待上，比真正调用还慢）
            self._apply_api_rate_limit(request_deque, _env_int("VL_RPM", 60))

            # 4.2 向模型发送请求
            summaries[img_file] = self._summarize_image(
                img_path, root_folder=doc_stem, image_content=context, chat_model=chat_model
            )

        return summaries

    def _summarize_image(
            self,
            image_path: str,
            root_folder: str,
            image_content: Tuple[str, str],
            chat_model=None,
    ) -> str:
        """
           调用多模态大模型总结图片内容。

           参数：
           - image_path: 图片本地路径。
           - root_folder: 文档所属文件夹名（提供更多上下文）。
           - image_content: 图片在文档中的上下文 (前文, 后文)。
           - chat_model: 复用的多模态客户端（不传则临时创建）。
        """

        # 1. 将图片转换成base64
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")


        try:
            # 走 vl 角色：多模态模型可以单独指向本地部署的 VL 模型
            if chat_model is None:
                chat_model = build_client(role="vl")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""这是"{root_folder}"文件中的一张图片，图片上文部分为"{image_content[0]}"，下文部分为"{image_content[1]}"，请用中文简要总结这张图片的内容，用于 Markdown 图片标题。"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]

            response = chat_model.invoke(messages)
            return response.content.strip().replace("\n","")

        except Exception as e:
            self.logger.error(f"获取图片摘要失败:{image_path}， 错误：{e}")
            # 失败时返回空串，不要返回 root_folder ——
            # 一份手册上百张图失败时，会在正文里塞进上百处重复的文档名，
            # 这些噪音会稀释 embedding，比留空更糟。
            return ""


    def _apply_api_rate_limit(
            self,
            request_times: Deque[float],
            max_requests: int,
            window_seconds: int = 60
    ) -> None:
        """
        通用滑动窗口API速率限制器（抽离为公共工具）
        核心逻辑：维护请求时间戳双端队列，窗口内请求数超上限则自动等待，防止触发第三方API限流
        :param request_times: 存储请求时间戳的双端队列，需外部初始化（全局/单例），跨调用复用
        :param max_requests: 速率限制窗口内的最大允许请求次数
        :param window_seconds: 速率限制滑动窗口时长，默认60秒（1分钟）
        :return: None，超出限制时会阻塞等待
        """
        # 1. 记录当前时间
        current_time = time.time()

        # 2. 清理滑动窗口中的过期请求
        while request_times and current_time - request_times[0] >= window_seconds:
            request_times.popleft()

        # 3. 窗口内请求数达到上限，计算需要等待的时间并阻塞
        if len(request_times) >= max_requests:
            #计算需要等待的时长 = 窗口的总时长 - 窗口内第一个请求的时间
            sleep_duration = window_seconds - (current_time - request_times[0])
            if sleep_duration > 0:
                self.logger.info(f"请求被限速，等待{sleep_duration:.2f}秒...")

                #等。。。。
                time.sleep(sleep_duration)

                #等完了
                current_time = time.time()
                while request_times and current_time - request_times[0] >= window_seconds:
                    request_times.popleft()

        # 4. 记录当前请求的时间戳，新请求入队
        request_times.append(current_time)
        self.logger.info(f"{self.name} 请求成功，当前{window_seconds}s窗口内请求次数为{len(request_times)}")



    def _step_2_scan_images(self, md_content: str, images_dir: Path) -> List[Tuple[str, str, Tuple[str, str]]]:
        """
        扫描图片文件夹，过滤出「支持格式+MD中实际引用」的图片，组装处理元数据
        :param md_content: MD文件完整内容
        :param images_dir: 图片文件夹路径对象
        :return: 待处理图片列表，每个元素为(图片文件名, 图片完整路径, 图片上下文)元组
        """

        target_images = []
        # 对图片文件夹进行遍历
        for image_file in os.listdir(images_dir):
            # 1. 过滤无效后缀
            file_ext = os.path.splitext(image_file)[1].lower()
            # 和合法后缀进行比较
            if file_ext not in self.config.image_extensions:
                self.logger.warning(f"图片{image_file}格式不支持")
                continue

            # 2. 组装图片的完整路径并转成字符串
            img_path = str(images_dir / image_file)

            # 3. 查找这个图片在md文档中引用的上下文
            context = self._find_image_in_md(md_content, image_file)

            if not context:
                self.logger.warning(f"图片{image_file}未在md文档中找到")
                continue

            # 4. 将查询到的图片组装到列表中
            target_images.append((image_file, img_path, context))

        return target_images

    def _step_1_get_content(self, state: ImportGraphState) -> Tuple[str, Path, Path]:
        """
        从全局状态中提取并初始化MD处理所需核心数据
        :param state: 流程全局状态对象
        :return: 元组(MD文件内容, MD文件路径, 图片文件夹路径)
        :raise FileProcessingError: 当状态中无有效MD文件路径时抛出
        """

        # 1. 参数非空校验
        md_path = state.get("md_path")
        if not md_path:
            raise StateFieldError(
                field_name="md_path",
                message="MD文件路径不能为空",
                expected_type=str)

        # 2. 路径转换
        md_path_obj = Path(md_path)

        # 3. 检查文件的有效性
        if not md_path_obj.exists():
            raise FileProcessingError(message=f"文件{md_path_obj.name}不存在")

        # 4. 获取md_content
        md_content = md_path_obj.read_text(encoding="utf-8")

        # 5. 获取md文件的图片文件夹路径
        img_dir = md_path_obj.parent / "images"

        return md_content, md_path_obj, img_dir

    def _find_image_in_md(
            self,
            md_content: str,  # md文件的完整内容
            image_file: str,  # 图片文件
            context_len: int = 100) -> Tuple[str, str]:

        # re.escape ： 给我将参数中的特殊符号进行转义
        # r：不要给我转义
        # 1 定义正则表达式
        #  ![](images/ac26d5ab3a9f599eb2f58c2f2cb89f009fd2172b49782804756ea10c7256d4b4.jpg)
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)")
        match = pattern.search(md_content)
        if not match:
            # 没有找到图片
            return None

        # 2 截取图片前后的上文和下文
        start, end = match.span()
        # print(f"start =  {start}")
        # print(f"end =  {end}")
        pre_text = md_content[max(0, start - context_len):start]
        post_text = md_content[end:min(len(md_content), end + context_len)]

        # 3 返回图片的前文和后文
        return pre_text, post_text


if __name__ == "__main__":
    setup_logging()

    md_path = r"D:\output\hak180产品安全手册\hak180产品安全手册.md"
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    init_state = {
        "md_path": md_path,
        "md_content": md_content
    }

    # 执行核心处理流程
    node_md_img = NodeMDImg()
    result = node_md_img(init_state)

    logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4))
