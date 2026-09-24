"""TLS cert generation for the phone server: must stay valid across a LAN IP change (DHCP lease renewal,
switching Wi-Fi networks) -- a cert whose SAN only lists a stale IP makes every browser refuse the connection
outright, which is exactly what happened when this machine's DHCP-assigned IP changed and the cached cert
(and the old, differently-resolved "Phone:" URL) still pointed at the old address."""
from __future__ import annotations

import ipaddress

import pytest

pytest.importorskip("cryptography")

from cryptography import x509

from cua.server.tls import ensure_cert, local_ip


def _cert_ips(cert_path) -> set:
    cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
    san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    return set(san.get_values_for_type(x509.IPAddress))


def test_ensure_cert_creates_cert_and_key(tmp_path):
    cert_path, key_path = ensure_cert(tmp_path)
    assert cert_path.is_file() and key_path.is_file()


def test_ensure_cert_includes_current_local_ip(tmp_path):
    cert_path, _ = ensure_cert(tmp_path)
    assert ipaddress.ip_address(local_ip()) in _cert_ips(cert_path)


def test_ensure_cert_reuses_existing_cert_when_ip_unchanged(tmp_path):
    cert_path, key_path = ensure_cert(tmp_path)
    first_bytes = cert_path.read_bytes()
    cert_path2, key_path2 = ensure_cert(tmp_path)
    assert cert_path2.read_bytes() == first_bytes      # not regenerated


def test_ensure_cert_regenerates_when_local_ip_changed(tmp_path, monkeypatch):
    cert_path, _ = ensure_cert(tmp_path)
    original_ips = _cert_ips(cert_path)
    assert ipaddress.ip_address("10.99.99.99") not in original_ips

    monkeypatch.setattr("cua.server.tls.local_ip", lambda: "10.99.99.99")
    cert_path2, _ = ensure_cert(tmp_path)
    new_ips = _cert_ips(cert_path2)
    assert ipaddress.ip_address("10.99.99.99") in new_ips     # cert was regenerated to cover the new IP


def test_local_ip_falls_back_to_loopback_on_error(monkeypatch):
    import socket as socket_mod

    class BrokenSocket:
        def __enter__(self):
            raise OSError("no network")

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(socket_mod, "socket", lambda *a, **k: BrokenSocket())
    assert local_ip() == "127.0.0.1"
