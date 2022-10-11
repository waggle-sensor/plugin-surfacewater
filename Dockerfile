FROM waggle/plugin-base:1.1.1-ml

COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY app.py /app/

ADD https://web.lcrc.anl.gov/public/waggle/models/surfacewater/surface_water_model_preliminary.pth /app/model.pth

WORKDIR /app
ENTRYPOINT ["python3", "/app/app.py"]
