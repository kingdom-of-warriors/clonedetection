# coding=utf-8
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import logging

from parser import DFG_python, DFG_java, DFG_ruby, DFG_go, DFG_php, DFG_javascript
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from tree_sitter import Language, Parser

logger = logging.getLogger(__name__)

# DFG函数映射
dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    'ruby': DFG_ruby,
    'go': DFG_go,
    'php': DFG_php,
    'javascript': DFG_javascript
}

class ParserManager:
    """管理不同语言的解析器"""
    def __init__(self):
        self.parsers = {}
        self._load_parsers()
    
    def _load_parsers(self):
        """加载所有语言的解析器"""
        for lang in dfg_function:
            LANGUAGE = Language('parser/my-languages.so', lang)
            parser = Parser()
            parser.set_language(LANGUAGE)
            parser = [parser, dfg_function[lang]]
            self.parsers[lang] = parser
    
    def get_parser(self, lang):
        """获取指定语言的解析器"""
        return self.parsers.get(lang)

def extract_dataflow(code, parser, lang):
    """提取代码的数据流信息"""
    # 移除注释
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
    
    # 获取数据流
    if lang == "php":
        code = "<?php" + code + "?>"
    
    try:
        tree = parser[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split('\n')
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)
        
        try:
            DFG, _ = parser[1](root_node, index_to_code, {})
        except:
            DFG = []
        
        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg = new_DFG
    except:
        dfg = []
    
    return code_tokens, dfg

class InputFeatures(object):
    """单个训练/测试样本的特征"""
    def __init__(self, input_tokens_1, input_ids_1, position_idx_1, dfg_to_code_1, dfg_to_dfg_1,
                 input_tokens_2, input_ids_2, position_idx_2, dfg_to_code_2, dfg_to_dfg_2,
                 label, url1, url2):
        # 第一个代码函数
        self.input_tokens_1 = input_tokens_1
        self.input_ids_1 = input_ids_1
        self.position_idx_1 = position_idx_1
        self.dfg_to_code_1 = dfg_to_code_1
        self.dfg_to_dfg_1 = dfg_to_dfg_1
        
        # 第二个代码函数
        self.input_tokens_2 = input_tokens_2
        self.input_ids_2 = input_ids_2
        self.position_idx_2 = position_idx_2
        self.dfg_to_code_2 = dfg_to_code_2
        self.dfg_to_dfg_2 = dfg_to_dfg_2
        
        # 标签
        self.label = label
        self.url1 = url1
        self.url2 = url2

class CodeProcessor:
    """代码处理器"""
    def __init__(self, parser_manager):
        self.parser_manager = parser_manager
    
    def process_code(self, code, tokenizer, args, lang='java'):
        """处理单个代码片段"""
        parser = self.parser_manager.get_parser(lang)
        
        # 提取数据流
        code_tokens, dfg = extract_dataflow(code, parser, lang)
        code_tokens = [tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x) 
                      for idx, x in enumerate(code_tokens)]
        
        ori2cur_pos = {}
        ori2cur_pos[-1] = (0, 0)
        for i in range(len(code_tokens)):
            ori2cur_pos[i] = (ori2cur_pos[i-1][1], ori2cur_pos[i-1][1] + len(code_tokens[i]))
        
        code_tokens = [y for x in code_tokens for y in x]
        
        # 截断
        code_tokens = code_tokens[:args.code_length + args.data_flow_length - 3 - min(len(dfg), args.data_flow_length)][:512-3]
        source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
        
        dfg = dfg[:args.code_length + args.data_flow_length - len(source_tokens)]
        source_tokens += [x[0] for x in dfg]
        position_idx += [0 for x in dfg]
        source_ids += [tokenizer.unk_token_id for x in dfg]
        
        padding_length = args.code_length + args.data_flow_length - len(source_ids)
        position_idx += [tokenizer.pad_token_id] * padding_length
        source_ids += [tokenizer.pad_token_id] * padding_length
        
        # 重新索引
        reverse_index = {}
        for idx, x in enumerate(dfg):
            reverse_index[x[1]] = idx
        
        for idx, x in enumerate(dfg):
            dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)
        
        dfg_to_dfg = [x[-1] for x in dfg]
        dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
        length = len([tokenizer.cls_token])
        dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]
        
        return source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg

def convert_examples_to_features(item):
    """将示例转换为特征"""
    url1, url2, label, tokenizer, args, cache, url_to_code, code_processor = item
    
    for url in [url1, url2]:
        if url not in cache:
            func = url_to_code[url]
            result = code_processor.process_code(func, tokenizer, args, 'java')
            cache[url] = result
    
    source_tokens_1, source_ids_1, position_idx_1, dfg_to_code_1, dfg_to_dfg_1 = cache[url1]
    source_tokens_2, source_ids_2, position_idx_2, dfg_to_code_2, dfg_to_dfg_2 = cache[url2]
    
    return InputFeatures(source_tokens_1, source_ids_1, position_idx_1, dfg_to_code_1, dfg_to_dfg_1,
                        source_tokens_2, source_ids_2, position_idx_2, dfg_to_code_2, dfg_to_dfg_2,
                        label, url1, url2)

class TextDataset(Dataset):
    """文本数据集"""
    def __init__(self, tokenizer, args, file_path='train'):
        self.examples = []
        self.args = args
        self.parser_manager = ParserManager()
        self.code_processor = CodeProcessor(self.parser_manager)
        
        self._load_data(tokenizer, args, file_path)
    
    def _load_data(self, tokenizer, args, file_path):
        """加载数据"""
        index_filename = file_path
        
        # 加载索引
        logger.info("Creating features from index file at %s", index_filename)
        url_to_code = {}
        with open('/'.join(index_filename.split('/')[:-1]) + '/data.jsonl') as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                url_to_code[js['idx']] = js['func']
        
        # 根据索引加载代码函数
        data = []
        cache = {}
        with open(index_filename) as f:
            for line in f:
                line = line.strip()
                url1, url2, label = line.split('\t')
                if url1 not in url_to_code or url2 not in url_to_code:
                    continue
                if label == '0':
                    label = 0
                else:
                    label = 1
                data.append((url1, url2, label, tokenizer, args, cache, url_to_code, self.code_processor))
        
        # 验证集只使用10%的数据
        if 'valid' in file_path:
            data = random.sample(data, int(len(data) * 0.1))
        
        # 转换示例为输入特征
        self.examples = [convert_examples_to_features(x) for x in tqdm(data, total=len(data))]
        
        if 'train' in file_path:
            self._log_examples()
    
    def _log_examples(self):
        """记录示例日志"""
        for idx, example in enumerate(self.examples[:3]):
            logger.info("*** Example ***")
            logger.info("idx: {}".format(idx))
            logger.info("label: {}".format(example.label))
            logger.info("input_tokens_1: {}".format([x.replace('\u0120', '_') for x in example.input_tokens_1]))
            logger.info("input_ids_1: {}".format(' '.join(map(str, example.input_ids_1))))
            logger.info("position_idx_1: {}".format(example.position_idx_1))
            logger.info("dfg_to_code_1: {}".format(' '.join(map(str, example.dfg_to_code_1))))
            logger.info("dfg_to_dfg_1: {}".format(' '.join(map(str, example.dfg_to_dfg_1))))
            
            logger.info("input_tokens_2: {}".format([x.replace('\u0120', '_') for x in example.input_tokens_2]))
            logger.info("input_ids_2: {}".format(' '.join(map(str, example.input_ids_2))))
            logger.info("position_idx_2: {}".format(example.position_idx_2))
            logger.info("dfg_to_code_2: {}".format(' '.join(map(str, example.dfg_to_code_2))))
            logger.info("dfg_to_dfg_2: {}".format(' '.join(map(str, example.dfg_to_dfg_2))))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, item):
        """获取单个样本"""
        return self._create_attention_masks(item)
    
    def _create_attention_masks(self, item):
        """创建注意力掩码"""
        # 为第一个代码创建图引导的掩码函数
        attn_mask_1 = np.zeros((self.args.code_length + self.args.data_flow_length,
                               self.args.code_length + self.args.data_flow_length), dtype=bool)
        
        # 计算节点开始索引和输入的最大长度
        node_index = sum([i > 1 for i in self.examples[item].position_idx_1])
        max_length = sum([i != 1 for i in self.examples[item].position_idx_1])
        
        # 序列可以关注序列
        attn_mask_1[:node_index, :node_index] = True
        
        # 特殊token关注所有token
        for idx, i in enumerate(self.examples[item].input_ids_1):
            if i in [0, 2]:
                attn_mask_1[idx, :max_length] = True
        
        # 节点关注从中识别的代码token
        for idx, (a, b) in enumerate(self.examples[item].dfg_to_code_1):
            if a < node_index and b < node_index:
                attn_mask_1[idx + node_index, a:b] = True
                attn_mask_1[a:b, idx + node_index] = True
        
        # 节点关注相邻节点
        for idx, nodes in enumerate(self.examples[item].dfg_to_dfg_1):
            for a in nodes:
                if a + node_index < len(self.examples[item].position_idx_1):
                    attn_mask_1[idx + node_index, a + node_index] = True
        
        # 为第二个代码创建图引导的掩码函数
        attn_mask_2 = np.zeros((self.args.code_length + self.args.data_flow_length,
                               self.args.code_length + self.args.data_flow_length), dtype=bool)
        
        # 计算节点开始索引和输入的最大长度
        node_index = sum([i > 1 for i in self.examples[item].position_idx_2])
        max_length = sum([i != 1 for i in self.examples[item].position_idx_2])
        
        # 序列可以关注序列
        attn_mask_2[:node_index, :node_index] = True
        
        # 特殊token关注所有token
        for idx, i in enumerate(self.examples[item].input_ids_2):
            if i in [0, 2]:
                attn_mask_2[idx, :max_length] = True
        
        # 节点关注从中识别的代码token
        for idx, (a, b) in enumerate(self.examples[item].dfg_to_code_2):
            if a < node_index and b < node_index:
                attn_mask_2[idx + node_index, a:b] = True
                attn_mask_2[a:b, idx + node_index] = True
        
        # 节点关注相邻节点
        for idx, nodes in enumerate(self.examples[item].dfg_to_dfg_2):
            for a in nodes:
                if a + node_index < len(self.examples[item].position_idx_2):
                    attn_mask_2[idx + node_index, a + node_index] = True
        
        return (torch.tensor(self.examples[item].input_ids_1),
                torch.tensor(self.examples[item].position_idx_1),
                torch.tensor(attn_mask_1),
                torch.tensor(self.examples[item].input_ids_2),
                torch.tensor(self.examples[item].position_idx_2),
                torch.tensor(attn_mask_2),
                torch.tensor(self.examples[item].label))