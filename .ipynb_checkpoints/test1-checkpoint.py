#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized RFdiffusion3 + ProteinMPNN + RF3 pipeline for CDR-H3 binder design
- 函数化每个大步骤
- 及时 del 模型 + torch.cuda.empty_cache()
- 只保留必要变量，减少显存驻留
- 支持单结构验证，易扩展批量
"""

import os
import json
import time
import torch
import numpy as np
from pathlib import Path
from lightning.fabric import seed_everything
from biotite.structure import get_residue_starts, rmsd, superimpose
from biotite.sequence import ProteinSequence
from atomworks.io.utils.io_utils import load_any, to_cif_file
from atomworks.io.utils.visualize import view  # 如果有可视化需求
from atomworks.constants import PROTEIN_BACKBONE_ATOM_NAMES

# ----------------------- 全局配置 -----------------------
os.environ["SHOULD_USE_CUEQUIVARIANCE"] = "0"
os.environ["DISABLE_CUEQUIVARIANCE"] = "1"
os.environ["TORCH_DTYPE"] = "fp16"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

seed_everything(0)

# 目录
BASE_DIR = Path("/home/alex/aidd/PDL1-4ZQK")
OUT_CDR = BASE_DIR / "out_cdr"
OUT_CIF = BASE_DIR / "out_cif"
for d in [OUT_CDR, OUT_CIF]:
    d.mkdir(exist_ok=True, parents=True)

# ----------------------- 工具函数 -----------------------
def clear_gpu():
    """释放显存"""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def print_mem():
    """打印当前显存使用"""
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Mem: {used:.2f} GB / {total:.2f} GB")

# ----------------------- 步骤1: RFD3 生成 backbone -----------------------
def run_rfd3(pdb_path: str, out_dir: Path, n_batches: int = 1, batch_size: int = 1) -> np.ndarray:
    from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine
    
    print_mem()
    spec = json.load(open(BASE_DIR / "protein_binder_design.json"))['pdl1_clean']
    config = RFD3InferenceConfig(specification=spec, diffusion_batch_size=batch_size)
    
    engine = RFD3InferenceEngine(**config)
    engine.run(inputs=pdb_path, out_dir=str(out_dir), n_batches=n_batches)
    del engine
    clear_gpu()
    print_mem()
    
    cif_files = sorted(out_dir.glob("pd_l1_clean*_model_*.cif.gz"))
    if not cif_files:
        raise FileNotFoundError("No CIF generated")
    
    atom_array = load_any(str(cif_files[0]), model=1)
    print(f"Loaded backbone: {cif_files[0].name} ({len(atom_array)} atoms)")
    return atom_array

# ----------------------- 步骤2: ProteinMPNN 序列设计 -----------------------
def run_mpnn(atom_array: np.ndarray, batch_size: int = 10) -> list:
    from mpnn.inference_engines.mpnn import MPNNInferenceEngine
    
    print_mem()
    engine_config = {
        "model_type": "protein_mpnn",
        "is_legacy_weights": True,
        "out_directory": None,
        "write_structures": False,
        "write_fasta": False,
    }
    input_configs = [{"batch_size": batch_size, "remove_waters": True}]
    
    model = MPNNInferenceEngine(**engine_config)
    outputs = model.run(input_dicts=input_configs, atom_arrays=[atom_array])
    del model
    clear_gpu()
    print_mem()
    
    # 打印序列
    print(f"Generated {len(outputs)} sequences:")
    for i, item in enumerate(outputs):
        res_starts = get_residue_starts(item.atom_array)
        seq = ''.join(ProteinSequence.convert_letter_3to1(res_name)
                      for res_name in item.atom_array.res_name[res_starts])
        print(f"Seq {i+1}: {seq}")
    
    return outputs

# ----------------------- 步骤3: RF3 结构验证 -----------------------
def run_rf3(atom_array: np.ndarray, example_id: str = "pdl1_clean") -> object:
    from rf3.inference_engines.rf3 import RF3InferenceEngine
    from rf3.utils.inference import InferenceInput
    
    print_mem()
    engine = RF3InferenceEngine(
        ckpt_path='rf3',
        verbose=False,
        n_recycles=1,
        diffusion_batch_size=1,
        num_steps=20
    )
    
    input_struct = InferenceInput.from_atom_array(atom_array, example_id=example_id)
    outputs = engine.run(inputs=input_struct)
    del engine
    clear_gpu()
    print_mem()
    
    key = list(outputs.keys())[0]
    print(f"Output keys: {outputs.keys()}")
    print(f"Models for '{key}': {len(outputs[key])}")
    
    rf3_out = outputs[key][0]
    print(f"RF3 atom_array: {len(rf3_out.atom_array)} atoms")
    print(f"Summary keys: {list(rf3_out.summary_confidences.keys())}")
    
    # Summary
    s = rf3_out.summary_confidences
    print("=== Summary ===")
    print(f" pLDDT: {s['overall_plddt']:.3f} | PAE: {s['overall_pae']:.2f} Å")
    print(f" pTM: {s['ptm']:.3f} | ipTM: {s.get('iptm', 'N/A')}")
    print(f" Rank score: {s['ranking_score']:.3f} | Clash: {s['has_clash']}")
    
    return rf3_out

# ----------------------- 步骤4: RMSD 验证 & 导出 -----------------------
def validate_and_export(generated: np.ndarray, refolded: np.ndarray, out_cif: Path):
    # 全结构 RMSD（参考用，通常高）
    bb_gen = generated[np.isin(generated.atom_name, PROTEIN_BACKBONE_ATOM_NAMES)]
    bb_ref = refolded[np.isin(refolded.atom_name, PROTEIN_BACKBONE_ATOM_NAMES)]
    bb_ref_fitted, _ = superimpose(bb_gen, bb_ref)
    rmsd_all = rmsd(bb_gen, bb_ref_fitted)
    print(f"全结构 Backbone RMSD: {rmsd_all:.2f} Å")
    
    # 只算 CDR-H3 (chain A)
    mask_a_gen = generated.chain_id == "A"
    mask_a_ref = refolded.chain_id == "A"
    bb_gen_cdr = generated[mask_a_gen][np.isin(generated[mask_a_gen].atom_name, PROTEIN_BACKBONE_ATOM_NAMES)]
    bb_ref_cdr = refolded[mask_a_ref][np.isin(refolded[mask_a_ref].atom_name, PROTEIN_BACKBONE_ATOM_NAMES)]
    
    if len(bb_gen_cdr) == len(bb_ref_cdr):
        bb_ref_fitted_cdr, _ = superimpose(bb_gen_cdr, bb_ref_cdr)
        rmsd_cdr = rmsd(bb_gen_cdr, bb_ref_fitted_cdr)
        print(f"CDR-H3 Backbone RMSD (chain A): {rmsd_cdr:.2f} Å")
        print("Interpretation:", "Excellent" if rmsd_cdr < 1.2 else "Good" if rmsd_cdr < 2.0 else "Moderate" if rmsd_cdr < 3.0 else "Poor")
    else:
        print("警告：CDR-H3 原子数不匹配")
    
    # 导出 CIF
    to_cif_file(generated, str(out_cif / "pdl1_generated.cif"))
    to_cif_file(refolded, str(out_cif / "pdl1_refolded.cif"))
    to_cif_file(generated[mask_a_gen], str(out_cif / "pdl1_generated_cdr.cif"))  # 只 chain A 的 generated CDR
    to_cif_file(refolded[mask_a_ref], str(out_cif / "pdl1_refolded_cdr.cif"))    # 只 chain A 的 refolded CDR

# ----------------------- 主流程 -----------------------
def main():
    # Step 1: RFD3
    atom_array = run_rfd3(
        pdb_path=str(BASE_DIR / "pd_l1_clean.pdb"),
        out_dir=OUT_CDR,
        n_batches=1,
        batch_size=1
    )
    
    # Step 2: MPNN
    mpnn_outputs = run_mpnn(atom_array, batch_size=10)
    
    # Step 3: RF3
    rf3_out = run_rf3(atom_array)
    
    # Step 4: 验证 & 导出
    validate_and_export(atom_array, rf3_out.atom_array, OUT_CIF)
    
    # 可选可视化
    # view(rf3_out.atom_array)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total time: {(time.time() - start_time)/60:.1f} min")