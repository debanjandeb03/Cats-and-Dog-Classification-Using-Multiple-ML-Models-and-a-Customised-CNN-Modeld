#Base Image
FROM python:3.11.13

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

#WorkDir                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
WORKDIR /app

#Copy
COPY . /app

#Run
RUN pip install -r requirements.txt

#Expose Port
EXPOSE 5000

#Command
CMD ["python","./app.py"]

