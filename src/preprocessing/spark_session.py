"""
Spark Session Management Module
Handles initialization, JVM configuration (Java 17 support), and clean shutdown of Apache Spark sessions.
"""

import os
import sys
import logging
from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)

# Standard JVM opens required for Java 17+ compatibility with Apache Spark
JAVA_17_OPTS = (
    "--add-opens=java.base/java.lang=ALL-UNNAMED "
    "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED "
    "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
    "--add-opens=java.base/java.io=ALL-UNNAMED "
    "--add-opens=java.base/java.net=ALL-UNNAMED "
    "--add-opens=java.base/java.nio=ALL-UNNAMED "
    "--add-opens=java.base/java.util=ALL-UNNAMED "
    "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED "
    "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED "
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
    "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED "
    "--add-opens=java.base/sun.security.action=ALL-UNNAMED "
    "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED "
    "--add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED"
)


def ensure_java_environment() -> None:
    """Ensure JAVA_HOME and Python worker paths are configured for PySpark execution on Windows."""
    # Ensure PySpark workers use the exact same Python executable as the current environment
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    if "JAVA_HOME" not in os.environ or not os.path.exists(os.environ["JAVA_HOME"]):
        potential_jdk_paths = [
            r"C:\Program Files\Eclipse Adoptium\jdk-17.0.21.9-hotspot",
            r"C:\Program Files\Java\jdk-17",
            r"C:\Program Files\Eclipse Adoptium\jdk-17",
            r"C:\Program Files\Microsoft\jdk-17",
        ]
        for path in potential_jdk_paths:
            if os.path.exists(path):
                os.environ["JAVA_HOME"] = path
                logger.info(f"Auto-configured JAVA_HOME -> {path}")
                break

    # Set PYSPARK_SUBMIT_ARGS to include Java 17 module parameters
    existing_submit_args = os.environ.get("PYSPARK_SUBMIT_ARGS", "")
    if "--add-opens" not in existing_submit_args:
        os.environ["PYSPARK_SUBMIT_ARGS"] = f'--driver-java-options "{JAVA_17_OPTS}" pyspark-shell'


def get_spark_session(app_name: str = "GlobalEventMarketAnalysis") -> SparkSession:
    """
    Creates or retrieves an active SparkSession with Windows and Java 17 settings.
    """
    ensure_java_environment()

    builder = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.extraJavaOptions", JAVA_17_OPTS)
        .config("spark.executor.extraJavaOptions", JAVA_17_OPTS)
        # Suppress verbose native IO warnings on Windows where winutils may not be installed
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem")
    )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def stop_spark_session(spark: SparkSession) -> None:
    """Safely stop an active SparkSession."""
    if spark is not None:
        try:
            spark.stop()
            logger.info("SparkSession stopped successfully.")
        except Exception as e:
            logger.warning(f"Error stopping SparkSession: {e}")
