pipeline {
    agent {
        docker { image 'python:3.7.6-slim' }
    }
    stages {
        stage('Checkout project') {
            steps {
                checkout(
                    [
                        $class: 'GitSCM',
                        branches: [[name: "*/master"]], 
                        doGenerateSubmoduleConfigurations: false, 
                        extensions: [[
                            $class: 'SubmoduleOption', 
                            disableSubmodules: false, 
                            parentCredentials: true, 
                            recursiveSubmodules: true, 
                            reference: '', 
                            trackingSubmodules: false
                        ]], 
                        submoduleCfg: [], 
                        userRemoteConfigs: [[credentialsId: 'astropath-github', url: 'https://github.com/astropathjhu/microscopealignment']]
                    ]
                )
            }
        }
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
