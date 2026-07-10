# ADR-021: Migrasi Basis Data ke PostgreSQL

## Status: Accepted

Penulis: sari

## Konteks

Layanan mikro `checkout` saat ini menggunakan Basis Data SQLite yang menyebabkan
kontensi saat beban tinggi. Kami memutuskan untuk memindahkan penyimpanan ke
PostgreSQL sebagai Basis Data utama.

## Komponen Terlibat

- Layanan mikro: `checkout`
- Basis Data: `postgresql`
- Jaringan: antarmuka internal via gRPC
- Pengembang: sari, budi

## Keputusan

Kami akan menggunakan `postgresql` sebagai Basis Data terdistribusi. API pembayaran
akan tetap menggunakan `stripe`. Metrik dikumpulkan via Prometheus.

## Dampak

Jika Basis Data gagal, maka layanan `checkout` akan menampilkan error 5xx.
Jaringan internal harus dikonfigurasi ulang. Pengembang bertanggung jawab atas
migrasi skema.
