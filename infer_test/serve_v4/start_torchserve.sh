rm logs/*

python mar_builder.py &&

torchserve --ncs --start --ts-config config.properties
