## Docker 化部署

```shell
cd my_spss
docker build -t my_spss:v1
docker run -d --name myspss -p 8002:8000 my_spss:v1
```
