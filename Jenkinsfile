pipeline {
    agent {
        docker { image 'python:3.7.6-slim' }
    }
    stages {
        stage('test printing stuff') {
            steps {
                echo "hi :)"
                sh 'ls'
            }
        }
    }
    post {
        always {
            publishHTML([
                reportDir: 'build/reports/tests/test',
                reportFiles: 'index.html', 
                reportName: 'test-report',
                alwaysLinkToLastBuild: false,
                keepAll: true,
                allowMissing: false
                ]
            )
        }
    }
}
