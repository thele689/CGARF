#!/usr/bin/env python3
"""
将OrcaLoca 300个实例从分散的文件夹形式转换为统一的JSONL格式

输入：input/Orcaloca/astropy__astropy-12907/searcher_*.json 等300个文件夹
输出：input/orcaloca.jsonl (统一的JSONL格式)
"""

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

def convert_orcaloca_to_jsonl():
    """转换OrcaLoca实例为JSONL格式"""
    
    orcaloca_dir = REPO_ROOT / "input" / "Orcaloca"
    output_file = REPO_ROOT / "input" / "orcaloca.jsonl"
    
    if not orcaloca_dir.exists():
        print(f"❌ 输入目录不存在: {orcaloca_dir}")
        return False
    
    # 获取所有实例文件夹
    instance_folders = sorted([
        d for d in orcaloca_dir.iterdir() 
        if d.is_dir() and "__" in d.name
    ])
    
    print(f"📁 找到 {len(instance_folders)} 个实例文件夹")
    
    # 写入JSONL文件
    with open(output_file, 'w') as out_f:
        count = 0
        failed = 0
        
        for instance_dir in instance_folders:
            instance_id = instance_dir.name
            
            try:
                # 查找searcher_*.json文件
                searcher_files = list(instance_dir.glob("searcher_*.json"))
                
                if not searcher_files:
                    print(f"⚠️  {instance_id}: 未找到searcher_*.json文件")
                    failed += 1
                    continue
                
                searcher_file = searcher_files[0]
                
                # 读取JSON数据
                with open(searcher_file, 'r') as f:
                    data = json.load(f)
                
                # 提取bug_locations
                bug_locations = data.get("bug_locations", [])
                
                # 提取repo信息
                # instance_id格式: "astropy__astropy-12907"
                # repo格式: "astropy/astropy"
                parts = instance_id.split("__")
                if len(parts) == 2:
                    owner = parts[0]
                    project_part = parts[1].rsplit("-", 1)[0]  # 去掉最后的数字
                    repo = f"{owner}/{project_part}"
                else:
                    repo = ""
                
                # 生成JSONL记录
                record = {
                    "instance_id": instance_id,
                    "repo": repo,
                    "bug_locations": bug_locations
                }
                
                # 写入JSONL
                json.dump(record, out_f)
                out_f.write('\n')
                
                count += 1
                if count % 50 == 0:
                    print(f"✓ 已处理 {count} 个实例...")
            
            except Exception as e:
                print(f"❌ 处理失败 {instance_id}: {e}")
                failed += 1
    
    print(f"\n✅ 转换完成!")
    print(f"   成功: {count} 个实例")
    print(f"   失败: {failed} 个实例")
    print(f"   输出文件: {output_file}")
    print(f"   文件大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    # 验证JSONL文件
    print(f"\n📋 验证JSONL格式...")
    with open(output_file, 'r') as f:
        first_line = f.readline()
        first_record = json.loads(first_line)
        print(f"   第一条记录: {first_record}")
    
    return True


if __name__ == "__main__":
    convert_orcaloca_to_jsonl()
