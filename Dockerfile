FROM python:3.9-buster

# RUN echo "deb https://mirrors.huaweicloud.com/debian/ buster main contrib non-free" > /etc/apt/sources.list && \
#     echo "deb https://mirrors.huaweicloud.com/debian/ buster-updates main contrib non-free" >> /etc/apt/sources.list && \
#     echo "deb https://mirrors.huaweicloud.com/debian-security/ buster/updates main contrib non-free" >> /etc/apt/sources.list

RUN apt-get update && \
    apt-get install -y gcc libc-dev default-mysql-client default-libmysqlclient-dev nginx libsasl2-dev libldap2-dev libssl-dev zip jq && \
    apt-get clean


RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn
    
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /app
COPY ./ /app/

CMD ["uvicorn", "claude_code_on_openai.server:app", "--host", "0.0.0.0", "--port", "8082", "--workers", "10"]
