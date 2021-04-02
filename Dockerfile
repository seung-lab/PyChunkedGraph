FROM gcr.io/neuromancer-seung-import/pychunkedgraph:graph-tool_dracopy
COPY override/timeout.conf /etc/nginx/conf.d/timeout.conf
COPY override/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY . /app
RUN pip install pip=20.2 \
    && pip install --no-cache-dir --upgrade -r requirements.txt
