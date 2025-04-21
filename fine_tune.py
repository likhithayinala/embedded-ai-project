
import sounddevice as sd
import soundfile as sf
import sys
sys.path.append('/content/Torch-KWT')

SAMPLE_RATE = 16000
DURATION = 1  # seconds


def fine_tune():
    # Record audio samples
    def record_clip(filename):
        print(f"Recording {filename} for {DURATION}s…")
        clip = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)  # sounddevice rec
        sd.wait()  # wait until recording is done
        sf.write(filename, clip, SAMPLE_RATE)

    # Record five “yes” samples
    for i in range(1, 6):
        record_clip(f"data/yes_{i}.wav")
    # (Optionally) record five “not_yes” samples
    for i in range(1, 6):
        record_clip(f"data/not_yes_{i}.wav")
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import torchaudio
    import torchaudio.transforms as T


    class YesNoDataset(Dataset):
        def __init__(self, file_list, label, transform):
            self.files = file_list
            self.label = label
            self.transform = transform

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            wav, sr = torchaudio.load(self.files[idx])
            if sr != SAMPLE_RATE:
                wav = T.Resample(sr, SAMPLE_RATE)(wav)  # resample if needed
            spec = T.MelSpectrogram(
                sample_rate=SAMPLE_RATE, n_fft=1024, win_length=640,
                hop_length=160, n_mels=40
            )(wav)  # (1, 40, T)
            log_mel = torch.log(spec + 1e-6)
            # pad/crop to 98 frames
            if log_mel.shape[-1] < 98:
                pad = 98 - log_mel.shape[-1]
                log_mel = F.pad(log_mel, (0, pad))
            log_mel = log_mel[:, :, :98]
            return log_mel, torch.tensor(self.label)

    # Assemble datasets
    yes_files = [f"data/yes_{i}.wav" for i in range(1,6)]
    no_files  = [f"data/not_yes_{i}.wav" for i in range(1,6)]
    transform = None  # already applied above

    dataset = torch.utils.data.ConcatDataset([
        YesNoDataset(yes_files, 1, transform),
        YesNoDataset(no_files, 0, transform)
    ])
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    import torch
    from kwt.models.kwt import kwt_from_name, KWT

    # 1) Load original 35-class model to grab weights
    base_model = kwt_from_name("kwt-1")  # default num_classes=35
    ckpt = torch.load("kwt/kwt1_v01.pth", map_location="cpu")  # your downloaded weights :contentReference[oaicite:8]{index=8}
    base_model.load_state_dict(ckpt, strict=False)

    # 2) Create new model for 2 classes
    cfg = {
        "input_res":[40,98],"patch_res":[40,1],"num_classes":2,
        "mlp_dim":256,"dim":64,"heads":1,"depth":12,
        "dropout":0.0,"emb_dropout":0.1,"pre_norm":False
    }
    model = KWT(**cfg)

    # 3) Transfer matching weights (all but mlp_head)
    state_dict = base_model.state_dict()
    # remove the old head weights so shapes match
    for key in list(state_dict):
        if key.startswith("mlp_head"):
            state_dict.pop(key)
    model.load_state_dict(state_dict, strict=False)  # ignore missing head :contentReference[oaicite:9]{index=9}
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    EPOCHS = 10
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)               # forward
            loss = criterion(logits, y)     # compute loss
            loss.backward()                 # backprop
            optimizer.step()                # update weights
            total_loss += loss.item()
        avg = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg:.4f}")
    torch.save(model.state_dict(), "yesno_trained_model.pth")  # save model weights
