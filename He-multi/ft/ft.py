from train import train_model

if __name__ == "__main__":
    _ = train_model(
        root="dataset",
        train_set="set.000",
        val_set="set.001",
        epochs=10000,
        batch_size=64,
        lr=1e-4,
        device="cpu",
        save_dir="checkpoints",
        pretrained_path=None,
        use_long_range=True,
        w_force=0.005,
        sr_descriptor_cuts=(4.0, 12.0),
        lr_descriptor_cuts=(4.0, 12.0),
    )
