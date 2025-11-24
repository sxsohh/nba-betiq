import pandas as pd
import glob

clean_dir = "data/clean"
files = glob.glob(f"{clean_dir}/*.csv")

for f in files:
    print("\nFixing", f)
    df = pd.read_csv(f, dtype=str)

    # Normalize column names
    cols = {c.lower(): c for c in df.columns}
    # find game_id column variants
    game_cols = [c for c in df.columns if c.lower() in ["gameid", "game_id", "game id", "game"]]

    if len(game_cols) == 0:
        print(" → No GAME_ID column found")
        continue

    original = game_cols[0]

    # Rename to GAME_ID
    df = df.rename(columns={original: "GAME_ID"})

    # Repair formatting
    df["GAME_ID"] = df["GAME_ID"].astype(str).str.replace(".0", "", regex=False)
    df["GAME_ID"] = df["GAME_ID"].str.zfill(10)

    # Save back
    df.to_csv(f, index=False)
    print(" → Fixed GAME_ID in", f)
