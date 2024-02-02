#!/bin/bash
mkdir -p "$HOME/tmp/setup-aws"
cd "$HOME/tmp/setup-aws"

# Download the CLI tool for linux
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscli-exe-linux-x86_64.zip"

# Import the amazon GPG public key
echo "
    -----BEGIN PGP PUBLIC KEY BLOCK-----

    mQINBF2Cr7UBEADJZHcgusOJl7ENSyumXh85z0TRV0xJorM2B/JL0kHOyigQluUG
    ZMLhENaG0bYatdrKP+3H91lvK050pXwnO/R7fB/FSTouki4ciIx5OuLlnJZIxSzx
    PqGl0mkxImLNbGWoi6Lto0LYxqHN2iQtzlwTVmq9733zd3XfcXrZ3+LblHAgEt5G
    TfNxEKJ8soPLyWmwDH6HWCnjZ/aIQRBTIQ05uVeEoYxSh6wOai7ss/KveoSNBbYz
    gbdzoqI2Y8cgH2nbfgp3DSasaLZEdCSsIsK1u05CinE7k2qZ7KgKAUIcT/cR/grk
    C6VwsnDU0OUCideXcQ8WeHutqvgZH1JgKDbznoIzeQHJD238GEu+eKhRHcz8/jeG
    94zkcgJOz3KbZGYMiTh277Fvj9zzvZsbMBCedV1BTg3TqgvdX4bdkhf5cH+7NtWO
    lrFj6UwAsGukBTAOxC0l/dnSmZhJ7Z1KmEWilro/gOrjtOxqRQutlIqG22TaqoPG
    fYVN+en3Zwbt97kcgZDwqbuykNt64oZWc4XKCa3mprEGC3IbJTBFqglXmZ7l9ywG
    EEUJYOlb2XrSuPWml39beWdKM8kzr1OjnlOm6+lpTRCBfo0wa9F8YZRhHPAkwKkX
    XDeOGpWRj4ohOx0d2GWkyV5xyN14p2tQOCdOODmz80yUTgRpPVQUtOEhXQARAQAB
    tCFBV1MgQ0xJIFRlYW0gPGF3cy1jbGlAYW1hem9uLmNvbT6JAlQEEwEIAD4CGwMF
    CwkIBwIGFQoJCAsCBBYCAwECHgECF4AWIQT7Xbd/1cEYuAURraimMQrMRnJHXAUC
    ZMKcEgUJCSEf3QAKCRCmMQrMRnJHXCilD/4vior9J5tB+icri5WbDudS3ak/ve4q
    XS6ZLm5S8l+CBxy5aLQUlyFhuaaEHDC11fG78OduxatzeHENASYVo3mmKNwrCBza
    NJaeaWKLGQT0MKwBSP5aa3dva8P/4oUP9GsQn0uWoXwNDWfrMbNI8gn+jC/3MigW
    vD3fu6zCOWWLITNv2SJoQlwILmb/uGfha68o4iTBOvcftVRuao6DyqF+CrHX/0j0
    klEDQFMY9M4tsYT7X8NWfI8Vmc89nzpvL9fwda44WwpKIw1FBZP8S0sgDx2xDsxv
    L8kM2GtOiH0cHqFO+V7xtTKZyloliDbJKhu80Kc+YC/TmozD8oeGU2rEFXfLegwS
    zT9N+jB38+dqaP9pRDsi45iGqyA8yavVBabpL0IQ9jU6eIV+kmcjIjcun/Uo8SjJ
    0xQAsm41rxPaKV6vJUn10wVNuhSkKk8mzNOlSZwu7Hua6rdcCaGeB8uJ44AP3QzW
    BNnrjtoN6AlN0D2wFmfE/YL/rHPxU1XwPntubYB/t3rXFL7ENQOOQH0KVXgRCley
    sHMglg46c+nQLRzVTshjDjmtzvh9rcV9RKRoPetEggzCoD89veDA9jPR2Kw6RYkS
    XzYm2fEv16/HRNYt7hJzneFqRIjHW5qAgSs/bcaRWpAU/QQzzJPVKCQNr4y0weyg
    B8HCtGjfod0p1A==
    =gdMc
    -----END PGP PUBLIC KEY BLOCK-----
" | sed -e 's|^ *||' > aws2.pub
cat aws2.pub
gpg --import aws2.pub

# Configure GPG to trust the key
#expect -c '
#spawn gpg --expert --pinentry-mode loopback --edit-key FB5DB77FD5C118B80511ADA8A6310ACC4672475C
#expect "gpg> "
#send "trust\r";
#expect "Your decision?"
#send "5\r";
#expect "Do you really want to set this key to ultimate trust? (y/N)"
#send "y\r";
#expect "gpg> "
#send "save\r";
#interact
#'

# Configure key to be trusted
# https://security.stackexchange.com/questions/129474/how-to-raise-a-key-to-ultimate-trust-on-another-machine
KEY_FPATH=aws2.pub
KEY_ID=$(gpg --list-packets <"$KEY_FPATH" | awk '$1=="keyid:"{print$2;exit}')
echo "KEY_ID = $KEY_ID"
(echo 5; echo y; echo save) |
  gpg --command-fd 0 --no-tty --no-greeting -q --edit-key "$KEY_ID" trust

# Download the signature and verify the CLI tool is signed by amazon
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip.sig" -o "awscli-exe-linux-x86_64.zip.sig"

gpg --verify awscli-exe-linux-x86_64.zip.sig awscli-exe-linux-x86_64.zip

# Unzip the downloaded installer
unzip -o awscli-exe-linux-x86_64.zip
# OR
# 7z x awscli-exe-linux-x86_64.zip

# If you want to install somewhere else, change the PREFIX variable
PREFIX="$HOME/.local"
mkdir -p "$PREFIX"/bin
./aws/install --install-dir "$PREFIX/aws-cli" --bin-dir "$PREFIX/bin" --update

"$PREFIX"/bin/aws --version
