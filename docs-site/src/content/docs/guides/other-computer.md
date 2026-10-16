---
title: Use the model from another computer
description: Share a model on your home network with hearthwork share and connect, with PIN pairing.
---

Run the model on one computer (say, a PC with a GPU) and use any supported agent from another one. Both need Hearthwork. The second computer needs no model, no llama.cpp and no setup.

```sh
hearthwork share           # on the computer with the model; leave it running
hearthwork connect         # on the other computer: finds it, then asks for the PIN shown on the first
hearthwork claude          # now uses the other computer's model (same for every agent, the menu, bench)
hearthwork disconnect      # back to local models
```

## Pairing and devices

- **Pairing.** `connect` asks to pair, and the host shows a 6-digit PIN to type on the other computer. The PIN works for 2 minutes, 3 wrong tries end the request, and too many wrong PINs lock pairing for 10 minutes. The host then gives the computer a long random key (it keeps only a hash), so later connections need no PIN.
- **Devices.** `hearthwork devices` lists trusted computers. `hearthwork devices remove <name>` revokes one at once.

## What the other computer can do

- **Same cleanup.** Requests are cleaned up on the host, so the other computer needs no model-specific setup. It uses whatever model the host runs and cannot start, stop or switch it.
- Use `hearthwork status` on the client to see which route it uses.

## Network notes

- **Windows host.** The firewall may ask to allow Python. Allow it on private networks. Sharing is refused while a network is marked Public: mark it Private, or pass `--allow-public`.
- **Ports.** Discovery uses UDP port 8485 and sharing uses port 8484. If discovery fails, run `hearthwork connect <ip>[:8484]`. See [Ports](/reference/ports-and-environment/).
- **Not encrypted on a home network.** The traffic is plain HTTP, so use it only on networks you trust. For anywhere-access with encryption, see [Tailscale](/guides/tailscale/).

If sharing or pairing fails, see [Common errors](/troubleshooting/common-errors/).
