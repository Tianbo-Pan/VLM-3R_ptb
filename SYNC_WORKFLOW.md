# VLM-3R / VLM-3R_ptb 同步命令总结

本文总结以下流程的关键命令：

1. 将本地 `VLM-3R` 中的内容同步到 `VLM-3R_ptb`
   - 排除 `CUT3R/`
   - 排除 `vcd/vcd_vision_token/results/`
   - 排除 `vcd/vcd_feature_degradation/results/`
2. 在 `VLM-3R_ptb` 中提交并 `git push` 到 GitHub
3. 在新服务器上把 GitHub 中的 `VLM-3R_ptb` 同步到 `/root/autodl-tmp/projects/VLM-3R_ptb`
4. 再将服务器上的 `VLM-3R_ptb` 覆盖同步到 `VLM-3R`

---

## 1. 本地：从 `VLM-3R` 同步到 `VLM-3R_ptb`

源目录：

```bash
/local_home/pantianbo/projects/vision_reasoning/VLM-3R
```

目标目录：

```bash
/local_home/pantianbo/projects/vision_reasoning/VLM-3R_ptb
```

### 1.1 先 dry-run 预览

```bash
rsync -avhn \
  --exclude='.git/' \
  --exclude='CUT3R/' \
  --exclude='vcd/vcd_vision_token/results/' \
  --exclude='vcd/vcd_feature_degradation/results/' \
  /local_home/pantianbo/projects/vision_reasoning/VLM-3R/ \
  /local_home/pantianbo/projects/vision_reasoning/VLM-3R_ptb/
```

### 1.2 确认无误后正式同步

```bash
rsync -avh \
  --exclude='.git/' \
  --exclude='CUT3R/' \
  --exclude='vcd/vcd_vision_token/results/' \
  --exclude='vcd/vcd_feature_degradation/results/' \
  /local_home/pantianbo/projects/vision_reasoning/VLM-3R/ \
  /local_home/pantianbo/projects/vision_reasoning/VLM-3R_ptb/
```

---

## 2. 本地：在 `VLM-3R_ptb` 中提交并推送到 GitHub

进入仓库：

```bash
cd /local_home/pantianbo/projects/vision_reasoning/VLM-3R_ptb
```

查看状态：

```bash
git status
git diff --stat
```

提交并推送：

```bash
git add -A
git commit -m "Sync from VLM-3R excluding CUT3R and VCD results"
git push origin main
```

如果远端比本地新，先 rebase 再 push：

```bash
git pull --rebase origin main
git push origin main
```

---

## 3. 新服务器：将 GitHub 上的 `VLM-3R_ptb` 同步到 `/root/autodl-tmp/projects/VLM-3R_ptb`

假设当前目录为：

```bash
cd /root/autodl-tmp/projects
```

### 3.1 如果服务器可以访问 GitHub

先拉取一个新的 `VLM-3R_ptb`：

```bash
rm -rf VLM-3R_ptb
git clone https://github.com/Tianbo-Pan/VLM-3R_ptb.git VLM-3R_ptb
```

如果是 private repo，推荐改用 SSH：

```bash
git clone git@github.com:Tianbo-Pan/VLM-3R_ptb.git VLM-3R_ptb
```

### 3.2 如果服务器无法访问 GitHub

如果服务器 `curl -I https://github.com` 超时，说明无法直接 clone。此时建议在本地打包/传输：

本地机器上：

```bash
cd /local_home/pantianbo/projects/vision_reasoning
tar -czf VLM-3R_ptb.tar.gz VLM-3R_ptb
scp VLM-3R_ptb.tar.gz root@<server_ip>:/root/autodl-tmp/projects/
```

服务器上解压：

```bash
cd /root/autodl-tmp/projects
tar -xzf VLM-3R_ptb.tar.gz
```

---

## 4. 新服务器：将 `VLM-3R_ptb` 覆盖同步到 `VLM-3R`

假设目录结构如下：

```bash
/root/autodl-tmp/projects/
├── VLM-3R
└── VLM-3R_ptb
```

### 4.1 先 dry-run 预览

```bash
cd /root/autodl-tmp/projects
rsync -avhn --exclude='.git/' VLM-3R_ptb/ VLM-3R/
```

### 4.2 正式同步

```bash
cd /root/autodl-tmp/projects
rsync -avh --exclude='.git/' VLM-3R_ptb/ VLM-3R/
```

这条命令的效果是：

- `VLM-3R` 本地独有文件：保留
- `VLM-3R_ptb` 有但 `VLM-3R` 没有的文件：补上
- 两边都有但内容不同的文件：以 `VLM-3R_ptb` 为准覆盖

---

## 5. 推荐执行顺序

### 本地同步并推送

```bash
rsync -avhn \
  --exclude='.git/' \
  --exclude='CUT3R/' \
  --exclude='vcd/vcd_vision_token/results/' \
  --exclude='vcd/vcd_feature_degradation/results/' \
  /local_home/pantianbo/projects/vision_reasoning/VLM-3R/ \
  /local_home/pantianbo/projects/vision_reasoning/VLM-3R_ptb/

rsync -avh \
  --exclude='.git/' \
  --exclude='CUT3R/' \
  --exclude='vcd/vcd_vision_token/results/' \
  --exclude='vcd/vcd_feature_degradation/results/' \
  /local_home/pantianbo/projects/vision_reasoning/VLM-3R/ \
  /local_home/pantianbo/projects/vision_reasoning/VLM-3R_ptb/

cd /local_home/pantianbo/projects/vision_reasoning/VLM-3R_ptb
git add -A
git commit -m "Sync from VLM-3R excluding CUT3R and VCD results"
git push origin main
```

### 服务器同步

```bash
cd /root/autodl-tmp/projects
rm -rf VLM-3R_ptb
git clone https://github.com/Tianbo-Pan/VLM-3R_ptb.git VLM-3R_ptb
rsync -avh --exclude='.git/' VLM-3R_ptb/ VLM-3R/
```

如果服务器无法访问 GitHub，则把 `git clone` 这一步替换为“本地打包 + `scp` 上传 + 服务器解压”。
