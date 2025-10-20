# =============================
# main.py2.0
# =============================
import argparse
import os
import sys
import importlib.util

ATIAS_LABELS_PATH = r"processed_data/Atias_Labels.txt"  # שינוי השם לקובץ ה־labels החדש

def _import_train_module():
    try:
        from train_classification import TrainConfig, train
        return TrainConfig, train
    except Exception as e1:
        root = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(root, 'train_classification.py'),
            os.path.join(root, 'Classification', 'train_classification.py'),
            os.path.join(root, 'classification', 'train_classification.py'),
        ]
        for f in candidates:
            if os.path.exists(f):
                spec = importlib.util.spec_from_file_location('train_classification', f)
                mod = importlib.util.module_from_spec(spec)
                sys.modules['train_classification'] = mod
                assert spec.loader is not None
                spec.loader.exec_module(mod)
                return mod.TrainConfig, mod.train
        try:
            from Classification.train_classification import TrainConfig, train
            return TrainConfig, train
        except Exception:
            tried = "\n".join(candidates)
            raise ModuleNotFoundError(
                "Could not locate 'train_classification.py'. "
                "Ensure the file exists and is named exactly 'train_classification.py'.\n"
                f"Tried:\n{tried}"
            ) from e1

TrainConfig, train = _import_train_module()

def parse_args():
    p = argparse.ArgumentParser(description="Train LFP word-classification model")
    p.add_argument('--data_path', type=str, default=r'processed_data/Patient_03_classification_data.npy')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--max_epochs', type=int, default=60)
    p.add_argument('--patience', type=int, default=10)
    # balancing & loss
    p.add_argument('--no_sampler', action='store_true', help='Disable WeightedRandomSampler (default: sampler ON)')
    p.add_argument('--focal', action='store_true', help='Use FocalLoss instead of CrossEntropy (default: CE)')
    p.add_argument('--gamma_focal', type=float, default=2.0)
    p.add_argument('--beta_cb', type=float, default=0.9999)
    p.add_argument('--tau_logit_adjust', type=float, default=0.0)
    # model toggles to match original by default
    p.add_argument('--use_mha', action='store_true', help='Enable Multi-Head Attention (default: off)')
    p.add_argument('--lstm_layers', type=int, default=1)
    p.add_argument('--conv_norm', type=str, default='batch', choices=['batch', 'group'])
    p.add_argument('--k1', type=int, default=15)
    p.add_argument('--k2', type=int, default=5)
    p.add_argument('--use_dilation', action='store_true')
    # misc
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--save_dir', type=str, default=r'Classification/models')
    p.add_argument('--save_name', type=str, default='best_cls_lstm.pth')
    return p.parse_args()

def _print_stage_headers(data_path: str, labels_path: str):
    print("----start of run stage----")
    print("----start of preprocessing stage----")
    print("===== Processing Patient_03 =====")
    print("[INFO] SKIP_LFP_EXPORT=True: skipping LFP→CSV export")
    print(f"[INFO] Data: {data_path.replace(os.sep,'/')}")
    print(f"[INFO] Labels file: {labels_path.replace(os.sep,'/')}")
    if not os.path.isfile(labels_path):
        print(f"[WARN] Labels file not found at: {labels_path.replace(os.sep,'/')}")
    print("----end of preprocessing stage----")
    print("----start of training Classification stage----")

def main():
    args = parse_args()
    cfg = TrainConfig(
        data_path=args.data_path,
        seed=args.seed,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        use_sampler=not args.no_sampler,              # default True
        use_focal=args.focal,                         # default False -> CE
        gamma_focal=args.gamma_focal,
        beta_cb=args.beta_cb,
        tau_logit_adjust=args.tau_logit_adjust,       # default 0.0
        grad_clip=args.grad_clip,
        save_dir=args.save_dir,
        save_name=args.save_name,
        # model defaults mimic your original best
        conv_norm=args.conv_norm,
        k1=args.k1,
        k2=args.k2,
        use_dilation=args.use_dilation,
        use_mha=args.use_mha,
        lstm_layers=args.lstm_layers,
    )

    # כותרות/לוגים בסגנון ההרצות שלך + הצגת נתיב ה־labels החדש
    _print_stage_headers(cfg.data_path, ATIAS_LABELS_PATH)

    ckpt_path = train(cfg)

    print("----end of training Classification stage----")
    print("----end of run stage----")
    print(f"\nSaved best checkpoint to: {os.path.normpath(ckpt_path)}")

if __name__ == "__main__":
    main()
