FROM rust AS rust_builder

RUN git clone --recurse-submodules https://github.com/jezberg/loandra.git && \
    cd loandra && \
    make

FROM python:3.12

WORKDIR /usr/src/prop_hunt
COPY . .
COPY --from=rust_builder loandra/loandra ./prop_hunt/

RUN pip install --no-cache-dir -r requirements.txt

