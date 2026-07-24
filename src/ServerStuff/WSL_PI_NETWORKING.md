# Raspberry Pi to WSL Server Networking

This project sends stereo camera frames from the Raspberry Pi to the Flask server
running in `ServerManager.py`.

When the server runs inside WSL, the Raspberry Pi cannot usually reach the WSL IP
directly. WSL sits behind a Windows virtual network, so the Pi must connect to the
Windows host's LAN IP, and Windows must forward that port into WSL.

## Symptoms

The cameras are detected and configured on the Pi, but vision stays empty:

```text
VISION= --.- cm | BBOX=(None)
```

The Pi cannot reach the WSL IP:

```bash
ping 172.20.202.128
curl -vk --connect-timeout 3 https://172.20.202.128:8443/health
```

Expected failure in this setup:

```text
100% packet loss
Connection timed out
```

The server still works locally inside WSL:

```bash
curl -k https://127.0.0.1:8443/health
```

Expected success:

```json
{"model_loaded":true,"opencv_gpu":"CPU only","status":"healthy"}
```

## Network Layout From The Working Case

WSL server IP:

```text
172.20.202.128
```

Windows Wi-Fi/LAN IP:

```text
10.33.106.212
```

Raspberry Pi IP:

```text
10.33.106.51
```

The Pi can reach `10.33.106.212`, but not `172.20.202.128`.

## Fix

Run this in **Admin PowerShell** on Windows.

Forward Windows port `8443` to the WSL server port:

```powershell
netsh interface portproxy add v4tov4 listenaddress=10.33.106.212 listenport=8443 connectaddress=172.20.202.128 connectport=8443
```

Allow inbound TCP traffic on port `8443`:

```powershell
New-NetFirewallRule -DisplayName "Vision Stick 8443" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8443
```

Verify the forwarding rule:

```powershell
netsh interface portproxy show v4tov4
```

Expected entry:

```text
10.33.106.212    8443    172.20.202.128    8443
```

## Pi Configuration

In the Pi client code, use the Windows host LAN IP and the correct port syntax:

```python
BASE_URL = "https://10.33.106.212:8443"
```

Use a colon before `8443`, not a slash.

Correct:

```text
https://10.33.106.212:8443
```

Wrong:

```text
https://10.33.106.212/8443
```

The slash version tries to use HTTPS port `443` and treats `8443` as a URL path.

## Test From The Pi

With `ServerManager.py` running in WSL:

```bash
curl -vk --connect-timeout 3 https://10.33.106.212:8443/health
```

Expected success:

```json
{"model_loaded":true,"opencv_gpu":"CPU only","status":"healthy"}
```

Then run:

```bash
python3 reacting_pi.py
```

## If WSL IP Changes

WSL IPs can change after reboot. Check the current WSL IP:

```bash
ip -br addr
```

Then update the portproxy rule from Admin PowerShell:

```powershell
netsh interface portproxy delete v4tov4 listenaddress=10.33.106.212 listenport=8443
netsh interface portproxy add v4tov4 listenaddress=10.33.106.212 listenport=8443 connectaddress=NEW_WSL_IP connectport=8443
```

## Common Mistake

Make sure the portproxy listen port is `8443`, not `844`.

Wrong:

```powershell
listenport=844
```

Correct:

```powershell
listenport=8443
```
