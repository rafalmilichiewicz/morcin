FROM mysql:8.4.0

# Copy init.sql into the image
COPY init.sql /docker-entrypoint-initdb.d/

# Check if init.sql exists
RUN test -e /docker-entrypoint-initdb.d/init.sql && { echo "init.sql found and copied successfully."; } || { echo "Error: init.sql not found!"; exit 1; }
