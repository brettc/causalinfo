from causalinfo import Variable, Equation, make_variables, equations
# TODO:
# if __name__ == '__main__':
#     import mappings
#
#     def f_and_or(i1, i2, o1, o2):
#         if i1 and i2:
#             o1[1] = 1.0
#         else:
#             o1[0] = 1.0
#         if i1 or i2:
#             o2[1] = 1.0
#         else:
#             o2[0] = 1.0
#
#     c, s, a, k = make_variables('C S A K', 2)
#     c.assign_uniform()
#     print c.assigned
#     e1 = Equation('e1', [c], [s, k], mappings.f_branch_same)
#     e2 = Equation('e2', [s], [a], mappings.f_same)
#     net = CausalNetwork([e1, e2])
#     j = net.generate_joint()
#     print j.probabilities
#     k.assign_uniform()
#     s.assign_uniform()
#     j = net.generate_joint(do=[k])
#     print j.mutual_info(k, a)
#     j = net.generate_joint(do=[s])
#     print j.probabilities
#     print j.mutual_info(s, a)


def test_variable_init():
    v = Variable('x', 3)
    print v.states


def test_equation_init():
    a, b, c = make_variables('a b c', 2)
    e1 = Equation('xor', [a, b], [c], equations.xor_)
    print e1.to_frame()
