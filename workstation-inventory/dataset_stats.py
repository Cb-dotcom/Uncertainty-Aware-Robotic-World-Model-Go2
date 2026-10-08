import sys, glob, os, numpy as np, pandas as pd
# 66 cols: state 0:45 (lin_vel 0:3, ang_vel 3:6, gravity 6:9, joint_pos 9:21, joint_vel 21:33, torque 33:45), action 45:57, contact 57:65, termination 65
print("file,rows,terms,terms_per_1k,tilted_frac(gz>-0.8),fallen_frac(gz>-0.3),mean_speed_xy,mean_abs_action")
for pat in sys.argv[1:]:
    for f in sorted(glob.glob(pat)):
        df = pd.read_csv(f, header=None, low_memory=False)
        if not np.issubdtype(df.dtypes.iloc[0], np.number):
            df = pd.read_csv(f, low_memory=False)
        a = df.to_numpy(dtype=np.float32)
        if a.shape[1] != 66: print(f"{f},SKIP cols={a.shape[1]}"); continue
        gz = a[:, 8]; v = np.linalg.norm(a[:, 0:2], axis=1); t = a[:, 65] > 0.5
        print(f"{os.path.relpath(f)},{len(a)},{int(t.sum())},{1000*t.mean():.3f},{(gz>-0.8).mean():.4f},{(gz>-0.3).mean():.4f},{v.mean():.3f},{np.abs(a[:,45:57]).mean():.3f}", flush=True)
