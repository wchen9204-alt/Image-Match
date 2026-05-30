#include "apps/registration_app.h"

// 程序入口仅负责将命令行控制权移交给应用层封装，避免在 main 中堆积业务逻辑。
int main(int argc, char** argv) {
    return ir::RegistrationApp::run(argc, argv);
}
