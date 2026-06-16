#pragma once

#include <string>

#include "core/context.h"
#include "core/types.h"

namespace ir {

/// ������ȡ���ӿڡ�
///
/// �����ͼ���м��ؼ��㲢���������ӣ����д�� `RegistrationContext`��
class IKeypointExtractor {
public:
    virtual ~IKeypointExtractor() = default;

    /// ������ȡ�����ƣ�������־�͵��������
    virtual std::string name() const = 0;

    /// ���ص�ǰ��ȡ����Ӧ���������͡�
    virtual KeypointType type() const = 0;

    /// ����������ƥ����ʹ�õľ������͡�
    virtual NormType normType() const = 0;

    /// ����������ִ����������������Ӽ��㡣
    virtual bool extract(RegistrationContext& ctx) = 0;
};

} // namespace ir

