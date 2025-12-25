#!/bin/bash

# =============================================================================
# 训练脚本示例集合
# 根据需要选择合适的脚本运行
# =============================================================================

# -----------------------------------------------------------------------------
# 示例 1: 从头开始训练
# -----------------------------------------------------------------------------
train_from_scratch() {
    echo "=== 从头开始训练 ==="
    python demo.py \
        --output ./workdirs/train_from_scratch \
        --learning_rate 1e-4 \
        --epoch_num 200 \
        --batch_size_train 4 \
        --batch_size_valid 1 \
        --model_save_fre 10
}

# -----------------------------------------------------------------------------
# 示例 2: 加载预训练权重继续训练（新的训练任务）
# -----------------------------------------------------------------------------
train_with_pretrained() {
    echo "=== 使用预训练权重开始新的训练 ==="
    python demo.py \
        --output ./workdirs/train_with_pretrained \
        --pretrained-weights ./workdirs/train_from_scratch/best.pth \
        --learning_rate 1e-4 \
        --epoch_num 200 \
        --batch_size_train 4
}

# -----------------------------------------------------------------------------
# 示例 3: 恢复中断的训练
# -----------------------------------------------------------------------------
resume_training() {
    echo "=== 恢复训练 ==="
    python demo.py \
        --output ./workdirs/my_experiment \
        --pretrained-weights ./workdirs/my_experiment/epoch_100.pth \
        --resume-training \
        --learning_rate 1e-4 \
        --epoch_num 200 \
        --batch_size_train 4
}

# -----------------------------------------------------------------------------
# 示例 4: 冻结encoder微调decoder
# -----------------------------------------------------------------------------
finetune_decoder() {
    echo "=== 冻结encoder，微调decoder ==="
    python demo.py \
        --output ./workdirs/finetune_decoder \
        --pretrained-weights ./workdirs/train_from_scratch/best.pth \
        --freeze-encoder \
        --learning_rate 5e-5 \
        --epoch_num 50 \
        --batch_size_train 4
}

# -----------------------------------------------------------------------------
# 示例 5: 完整训练流程（包含mask cache）
# -----------------------------------------------------------------------------
train_with_mask_cache() {
    echo "=== 使用mask cache训练 ==="
    python demo.py \
        --output ./workdirs/train_with_cache \
        --pretrained-weights ./workdirs/train_from_scratch/best.pth \
        --use_mask_cache True \
        --update_mask_cache True \
        --learning_rate 1e-4 \
        --epoch_num 200 \
        --batch_size_train 4
}

# -----------------------------------------------------------------------------
# 示例 6: 严格加载模式
# -----------------------------------------------------------------------------
train_strict_load() {
    echo "=== 严格加载模式（所有keys必须匹配） ==="
    python demo.py \
        --output ./workdirs/train_strict \
        --pretrained-weights ./workdirs/train_from_scratch/best.pth \
        --load-strict \
        --learning_rate 1e-4 \
        --epoch_num 200 \
        --batch_size_train 4
}

# -----------------------------------------------------------------------------
# 示例 7: 评估模式
# -----------------------------------------------------------------------------
evaluate_model() {
    echo "=== 评估模型 ==="
    python demo.py \
        --output ./workdirs/evaluation \
        --restore-model ./workdirs/train_from_scratch/best.pth \
        --eval \
        --batch_size_valid 1
}

# -----------------------------------------------------------------------------
# 示例 8: 迁移学习（从其他数据集的模型开始）
# -----------------------------------------------------------------------------
transfer_learning() {
    echo "=== 迁移学习 ==="
    python demo.py \
        --output ./workdirs/transfer_learning \
        --pretrained-weights ./pretrained_models/other_dataset_best.pth \
        --learning_rate 5e-5 \
        --epoch_num 100 \
        --batch_size_train 4
}

# -----------------------------------------------------------------------------
# 示例 9: 两阶段训练（先冻结后解冻）
# -----------------------------------------------------------------------------
two_stage_training() {
    echo "=== 两阶段训练 ==="
    
    # 阶段1: 冻结encoder，训练decoder
    echo "--- 阶段1: 训练decoder ---"
    python demo.py \
        --output ./workdirs/two_stage \
        --pretrained-weights ./pretrained_models/pretrained.pth \
        --freeze-encoder \
        --learning_rate 1e-4 \
        --epoch_num 50 \
        --batch_size_train 4
    
    # 阶段2: 解冻所有层，微调整个网络
    echo "--- 阶段2: 微调整个网络 ---"
    python demo.py \
        --output ./workdirs/two_stage \
        --pretrained-weights ./workdirs/two_stage/best.pth \
        --resume-training \
        --learning_rate 1e-5 \
        --epoch_num 100 \
        --batch_size_train 4
}

# -----------------------------------------------------------------------------
# 示例 10: 快速测试（小epoch数）
# -----------------------------------------------------------------------------
quick_test() {
    echo "=== 快速测试 ==="
    python demo.py \
        --output ./workdirs/quick_test \
        --pretrained-weights ./workdirs/train_from_scratch/best.pth \
        --learning_rate 1e-4 \
        --epoch_num 5 \
        --batch_size_train 2 \
        --model_save_fre 1
}

# =============================================================================
# 主菜单
# =============================================================================
show_menu() {
    echo ""
    echo "================================================"
    echo "训练脚本示例菜单"
    echo "================================================"
    echo "1)  从头开始训练"
    echo "2)  使用预训练权重开始新训练"
    echo "3)  恢复中断的训练"
    echo "4)  冻结encoder微调decoder"
    echo "5)  使用mask cache训练"
    echo "6)  严格加载模式"
    echo "7)  评估模型"
    echo "8)  迁移学习"
    echo "9)  两阶段训练"
    echo "10) 快速测试"
    echo "q)  退出"
    echo "================================================"
    echo -n "请选择 (1-10 或 q): "
}

# =============================================================================
# 运行选择的脚本
# =============================================================================
if [ $# -eq 0 ]; then
    # 交互模式
    while true; do
        show_menu
        read choice
        case $choice in
            1) train_from_scratch ;;
            2) train_with_pretrained ;;
            3) resume_training ;;
            4) finetune_decoder ;;
            5) train_with_mask_cache ;;
            6) train_strict_load ;;
            7) evaluate_model ;;
            8) transfer_learning ;;
            9) two_stage_training ;;
            10) quick_test ;;
            q|Q) echo "退出"; exit 0 ;;
            *) echo "无效选择，请重试" ;;
        esac
        echo ""
        echo "按Enter继续..."
        read
    done
else
    # 命令行模式
    case $1 in
        scratch) train_from_scratch ;;
        pretrained) train_with_pretrained ;;
        resume) resume_training ;;
        finetune) finetune_decoder ;;
        cache) train_with_mask_cache ;;
        strict) train_strict_load ;;
        eval) evaluate_model ;;
        transfer) transfer_learning ;;
        twostage) two_stage_training ;;
        test) quick_test ;;
        *)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  scratch     - 从头开始训练"
            echo "  pretrained  - 使用预训练权重"
            echo "  resume      - 恢复训练"
            echo "  finetune    - 微调decoder"
            echo "  cache       - 使用mask cache"
            echo "  strict      - 严格加载"
            echo "  eval        - 评估模型"
            echo "  transfer    - 迁移学习"
            echo "  twostage    - 两阶段训练"
            echo "  test        - 快速测试"
            echo ""
            echo "不带参数运行以使用交互式菜单"
            exit 1
            ;;
    esac
fi

