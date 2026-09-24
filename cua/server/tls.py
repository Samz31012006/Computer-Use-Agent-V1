"""Self-signed TLS certificate for the local phone server.

Browsers require a "secure context" (HTTPS, or localhost) before they'll grant microphone access or expose the
Web Speech API -- plain http://<lan-ip>:8420 is neither, which is why the phone previously reported "No
microphone access" regardless of permissions. A CA-signed cert isn't possible for a private LAN IP, so this
generates a self-signed one instead: the phone's browser will show a one-time "not secure" / "proceed anyway"
warning (there is no CA to vouch for it), but the connection IS encrypted after that, which is what satisfies
the browser's secure-context check and stops anyone else on the LAN from passively reading the traffic.

The key/cert pair is generated once and cached on disk (.cache/), not regenerated per run.
"""
from __future__ import annotations

import datetime
import socket
from pathlib import Path


def local_ip() -> str:
    """The IP address this machine would actually use to reach the LAN -- found by asking the OS routing
    table which local interface it would pick for an outbound connection (no packet is actually sent; UDP
    `connect()` just resolves a route). This is the SAME method used to build the cert's SAN below, and is
    also what the phone URL is printed from (cua/server/api.py) -- previously that print used
    `socket.gethostbyname(socket.gethostname())`, which on Windows can return a stale or entirely different
    address (a VPN adapter, a cached DNS entry, an old DHCP lease) than the interface actually serving this
    HTTP server, so the cert and the printed URL could silently disagree, or the URL could just be wrong."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def ensure_cert(cache_dir: str | Path = ".cache") -> tuple[Path, Path]:
    """Returns (cert_path, key_path), generating a self-signed cert on first call. Requires the `cryptography`
    package; raises ImportError if it isn't installed (callers should fall back to plain HTTP with a clear
    warning rather than silently failing).

    Regenerates the cert if the machine's current LAN IP isn't already covered by it -- a laptop's DHCP
    lease can change between runs (or even within one, on Wi-Fi), and a cert whose SAN only lists the OLD
    IP makes every browser refuse the connection outright once the phone is pointed at the new address."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    import ipaddress

    d = Path(cache_dir)
    cert_path, key_path = d / "server.crt", d / "server.key"
    ip = local_ip()
    if cert_path.is_file() and key_path.is_file():
        try:
            cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
            not_expired = cert.not_valid_after_utc > datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1)
            san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
            covers_current_ip = ipaddress.ip_address(ip) in san.get_values_for_type(x509.IPAddress)
            if not_expired and covers_current_ip:
                return cert_path, key_path
        except Exception:
            pass                                      # unreadable/expired/IP-changed: regenerate below

    d.mkdir(parents=True, exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "buddy.local")])
    san_names = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
    try:
        san_names.append(x509.IPAddress(ipaddress.ip_address(ip)))
    except ValueError:
        pass
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder()
           .subject_name(subject).issuer_name(issuer).public_key(key.public_key())
           .serial_number(x509.random_serial_number())
           .not_valid_before(now - datetime.timedelta(days=1))
           .not_valid_after(now + datetime.timedelta(days=825))
           .add_extension(x509.SubjectAlternativeName(san_names), critical=False)
           .sign(key, hashes.SHA256()))

    key_path.write_bytes(key.private_bytes(encoding=serialization.Encoding.PEM,
                                           format=serialization.PrivateFormat.PKCS8,
                                           encryption_algorithm=serialization.NoEncryption()))
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return cert_path, key_path


def tls_available() -> bool:
    try:
        import cryptography  # noqa: F401
        return True
    except ImportError:
        return False
