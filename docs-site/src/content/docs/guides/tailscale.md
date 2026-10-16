---
title: From anywhere with Tailscale
description: Reach your model from outside your home network over an encrypted Tailscale network.
---

[Tailscale](https://tailscale.com) is a WireGuard-based private network between your own devices. It is optional. Without it, nothing changes.

Install Tailscale and sign in on both computers, then:

```sh
hearthwork share --tailscale   # on the model computer
hearthwork connect             # on the other one: lists LAN hosts and Tailscale peers "(tailscale)"
```

## Host

`share` prints the Tailscale address and the MagicDNS name. With `--tailscale`, the host accepts only Tailscale and local computers (plus your local network when no network is Public).

## Client

`connect` also asks every online Tailscale peer, so you do not need to know an address. Pairing stores every address of the host. Each run tries the local network first, then Tailscale, so the same key works at home and away. `hearthwork status` shows the route in use.

## Security

Tailscale traffic is encrypted end to end, and only devices of your tailnet can reach the host. The PIN and device keys still apply, because a tailnet can be shared with other people. See [Use the model from another computer](/guides/other-computer/) for how pairing works.
