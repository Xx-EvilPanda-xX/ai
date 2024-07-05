// Args to each functor are its inputs
// multiplication of only two symbols at once
#[derive(Clone, Debug)]
pub enum Function {
    Add(Vec<Symbol>),
    Mul(Box<(Symbol, Symbol)>),
    Sig(Box<Symbol>),
    Inv(Box<Symbol>),
    Const(f64)
}

#[derive(Clone, Debug)]
pub struct Variable {
    pub index: usize
}

#[derive(Clone, Debug)]
pub enum Symbol {
    Func(Function),
    Var(Variable),
}

impl Symbol {
    pub fn to_closure<'a>(&'a self) -> impl Fn(&[f64]) -> f64 + 'a {
        move |inputs| {
            match self {
                Symbol::Func(f) => {
                    match f {
                        Function::Add(next) => next.iter().map(|symbol| symbol.to_closure()(inputs)).sum(),
                        Function::Mul(next) => next.0.to_closure()(inputs) * next.1.to_closure()(inputs),
                        Function::Sig(next) => 1.0 / (1.0 + (-next.to_closure()(inputs)).exp()),
                        Function::Inv(next) => 1.0 / next.to_closure()(inputs),
                        Function::Const(value) => *value,
                    }
                }
                Symbol::Var(v) => {
                    inputs[v.index]
                }
            }
        }
    }

    // TODO: OPTMIZEEEEEEE
    pub fn derivative(&self, respect: usize) -> Self {
        match self {
            Symbol::Func(f) => {
                match f {
                    Function::Add(next) => {
                        Symbol::Func(Function::Add(
                            next.iter().map(|symbol| symbol.derivative(respect)).collect()
                        ))
                    },
                    Function::Mul(next) => {
                        Symbol::Func(Function::Add(
                            vec![
                                Symbol::Func(Function::Mul(Box::new((
                                    next.0.clone(),
                                    next.1.derivative(respect)
                                )))),
                                Symbol::Func(Function::Mul(Box::new((
                                    next.1.clone(),
                                    next.0.derivative(respect)
                                )))),
                            ]
                        ))
                    },
                    Function::Sig(next) => {
                        Symbol::Func(Function::Mul(Box::new((
                            Symbol::Func(Function::Mul(Box::new((
                                Symbol::Func(Function::Mul(Box::new((
                                    Symbol::Func(Function::Add(
                                        vec![
                                            Symbol::Func(Function::Inv(Box::new(Symbol::Func(Function::Sig(next.clone()))))),
                                            Symbol::Func(Function::Const(-1.0))
                                        ]
                                    )),
                                    next.derivative(respect)
                                )))),
                                Symbol::Func(Function::Sig(next.clone()))
                            )))),
                            Symbol::Func(Function::Sig(next.clone()))
                        ))))
                    },
                    Function::Inv(next) => {
                        Symbol::Func(Function::Mul(Box::new((
                            next.derivative(respect),
                            Symbol::Func(Function::Inv(Box::new(
                                Symbol::Func(Function::Mul(Box::new((
                                    Symbol::Func(Function::Mul(Box::new((Symbol::Func(Function::Const(-1.0)), next.as_ref().clone())))),
                                    next.as_ref().clone()
                                ))))
                            )))
                        ))))
                    },
                    Function::Const(_) => Symbol::Func(Function::Const(0.0)),
                }
            },
            Symbol::Var(var) => {
                if var.index == respect {
                    Symbol::Func(Function::Const(1.0))
                } else {
                    Symbol::Func(Function::Const(0.0))
                }
            }
        }
    }
}

#[test]
fn test_function() {
    let s = Symbol::Func(
        Function::Add(
            vec![
                Symbol::Func(Function::Mul(Box::new((
                    Symbol::Func(Function::Const(5.0)), Symbol::Var(Variable{ index: 0 })
                )))),
                Symbol::Func(Function::Mul(Box::new((
                    Symbol::Func(Function::Const(2.0)), Symbol::Var(Variable{ index: 1 })
                )))),
                Symbol::Func(Function::Mul(Box::new((
                    Symbol::Func(Function::Const(7.0)), Symbol::Var(Variable{ index: 2 })
                ))))
            ]
        )
    );

    let inputs = [3.0, 2.0, 1.0];
    let closure = s.to_closure();

    assert_eq!(closure(&inputs), 26.0)
}

#[test]
fn test_derivative() {
    let s = Symbol::Func(
        Function::Add(
            vec![
                Symbol::Func(Function::Mul(Box::new((
                    Symbol::Var(Variable { index: 0 }),
                    Symbol::Var(Variable { index: 0 })
                )))),
                Symbol::Func(Function::Mul(Box::new((
                    Symbol::Func(Function::Const(3.0)),
                    Symbol::Func(Function::Inv(Box::new(
                        Symbol::Func(Function::Mul(Box::new((
                            Symbol::Var(Variable { index: 0 }),
                            Symbol::Var(Variable { index: 1 }),
                        ))))
                    )))
                )))),
                Symbol::Func(Function::Mul(Box::new((
                    Symbol::Var(Variable { index: 2 }),
                    Symbol::Func(Function::Sig(Box::new(
                        Symbol::Var(Variable { index: 0 })
                    )))
                ))))
            ]
        )
    );

    
    let der = s.derivative(0);
    let closure = der.to_closure();
    
    let inputs = [5.0, 4.8, -3.6];

    assert_eq!(closure(&inputs), 9.951066995985155);

    let inputs = [0.5, 2.3, 7.6];

    assert_eq!(closure(&inputs), -2.431363091615708);
}
