pub struct TriStore<T> {
    inner: TriStoreEnum<T>,
}

impl<T> TriStore<T> {
    pub fn take(&mut self) -> Self {
        let inner = std::mem::take(&mut self.inner);
        Self { inner }
    }

    pub const fn good(&self) -> Option<&T> {
        self.inner.good()
    }

    pub const fn good_mut(&mut self) -> Option<&mut T> {
        self.inner.good_mut()
    }

    pub const fn bad(&mut self) -> Option<&T> {
        self.inner.bad()
    }

    pub const fn bad_mut(&mut self) -> Option<&mut T> {
        self.inner.bad_mut()
    }

    pub const fn any(&self) -> Option<&Quality<T>> {
        self.inner.any()
    }

    pub const fn any_mut(&mut self) -> Option<&mut Quality<T>> {
        self.inner.any_mut()
    }

    pub fn good_or_set<F>(&mut self, f: F) -> &mut T
    where
        F: FnOnce() -> T,
    {
        self.inner.good_or_set(f)
    }

    pub fn bad_or_set<F>(&mut self, f: F) -> &mut T
    where
        F: FnOnce() -> T,
    {
        self.inner.bad_or_set(f)
    }

    pub fn any_or_set_good<F>(&mut self, f: F) -> &mut Quality<T>
    where
        F: FnOnce() -> T,
    {
        self.inner.any_or_set_good(f)
    }

    pub fn any_or_set_bad<F>(&mut self, f: F) -> &mut Quality<T>
    where
        F: FnOnce() -> T,
    {
        self.inner.any_or_set_bad(f)
    }

    pub fn any_or_try_set_good<F>(&mut self, f: F) -> Option<&mut Quality<T>>
    where
        F: FnOnce() -> Option<T>,
    {
        self.inner.any_or_try_set_good(f)
    }

    pub fn any_or_try_set_bad<F>(&mut self, f: F) -> Option<&mut Quality<T>>
    where
        F: FnOnce() -> Option<T>,
    {
        self.inner.any_or_try_set_bad(f)
    }

    pub fn discard(&mut self) {
        self.inner.discard()
    }

    pub fn degrade(&mut self) {
        self.inner.degrade()
    }

    pub const fn keep(&self) {}

    pub fn state(&self) -> TriState {
        self.inner.state()
    }
}

impl<T> Default for TriStore<T> {
    fn default() -> Self {
        Self {
            inner: Default::default(),
        }
    }
}

impl<T> std::fmt::Debug for TriStore<T>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt(f)
    }
}

#[derive(Default)]
enum TriStoreEnum<T> {
    #[default]
    Missing,
    Present(Quality<T>),
}

impl<T> TriStoreEnum<T> {
    fn take(&mut self) -> Self {
        std::mem::take(self)
    }

    const fn good(&self) -> Option<&T> {
        match self {
            Self::Missing => None,
            Self::Present(q) => q.good(),
        }
    }

    const fn good_mut(&mut self) -> Option<&mut T> {
        match self {
            Self::Missing => None,
            Self::Present(q) => q.good_mut(),
        }
    }

    const fn bad(&mut self) -> Option<&T> {
        match self {
            Self::Missing => None,
            Self::Present(q) => q.bad(),
        }
    }

    const fn bad_mut(&mut self) -> Option<&mut T> {
        match self {
            Self::Missing => None,
            Self::Present(q) => q.bad_mut(),
        }
    }

    const fn any(&self) -> Option<&Quality<T>> {
        match self {
            Self::Missing => None,
            Self::Present(quality) => Some(quality),
        }
    }

    const fn any_mut(&mut self) -> Option<&mut Quality<T>> {
        match self {
            Self::Missing => None,
            Self::Present(quality) => Some(quality),
        }
    }

    fn good_or_set<F>(&mut self, f: F) -> &mut T
    where
        F: FnOnce() -> T,
    {
        match self {
            Self::Present(q) if q.good().is_some() => (),
            Self::Missing | Self::Present(_) => *self = Self::Present(Quality::new_good(f())),
        }
        self.good_mut().expect("set above")
    }

    fn bad_or_set<F>(&mut self, f: F) -> &mut T
    where
        F: FnOnce() -> T,
    {
        match self {
            Self::Present(q) if q.bad().is_some() => (),
            Self::Missing | Self::Present(_) => *self = Self::Present(Quality::new_bad(f())),
        }
        self.bad_mut().expect("set above")
    }

    fn any_or_set_good<F>(&mut self, f: F) -> &mut Quality<T>
    where
        F: FnOnce() -> T,
    {
        match self {
            Self::Missing => *self = Self::Present(Quality::new_good(f())),
            Self::Present(_) => (),
        }
        self.any_mut().expect("set above")
    }

    fn any_or_set_bad<F>(&mut self, f: F) -> &mut Quality<T>
    where
        F: FnOnce() -> T,
    {
        match self {
            Self::Present(_) => (),
            Self::Missing => *self = Self::Present(Quality::new_bad(f())),
        }
        self.any_mut().expect("set above")
    }

    fn any_or_try_set_good<F>(&mut self, f: F) -> Option<&mut Quality<T>>
    where
        F: FnOnce() -> Option<T>,
    {
        match self {
            Self::Missing if let Some(value) = f() => {
                *self = Self::Present(Quality::new_good(value))
            }
            Self::Missing => (),
            Self::Present(_) => (),
        }
        self.any_mut()
    }

    fn any_or_try_set_bad<F>(&mut self, f: F) -> Option<&mut Quality<T>>
    where
        F: FnOnce() -> Option<T>,
    {
        match self {
            Self::Missing if let Some(value) = f() => {
                *self = Self::Present(Quality::new_bad(value))
            }
            Self::Missing => (),
            Self::Present(_) => (),
        }
        self.any_mut()
    }

    fn discard(&mut self) {
        self.take();
    }

    fn degrade(&mut self) {
        match self {
            Self::Missing => (),
            Self::Present(q) => q.degrade(),
        }
    }

    fn state(&self) -> TriState {
        match self {
            Self::Missing => TriState::Missing,
            Self::Present(q) => q.state(),
        }
    }
}

impl<T> std::fmt::Debug for TriStoreEnum<T>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing => write!(f, "Missing"),
            Self::Present(arg0) => f.debug_tuple("Present").field(arg0).finish(),
        }
    }
}

pub struct Quality<T> {
    inner: QualityEnum<T>,
}

impl<T> Quality<T> {
    pub fn new_good(t: T) -> Self {
        let inner = QualityEnum::Good(t);
        Self { inner }
    }

    pub fn new_bad(t: T) -> Self {
        let inner = QualityEnum::Bad(t);
        Self { inner }
    }

    pub const fn good(&self) -> Option<&T> {
        self.inner.good()
    }

    pub const fn good_mut(&mut self) -> Option<&mut T> {
        self.inner.good_mut()
    }

    pub const fn bad(&self) -> Option<&T> {
        self.inner.bad()
    }

    pub const fn bad_mut(&mut self) -> Option<&mut T> {
        self.inner.bad_mut()
    }

    pub fn or_update<F>(&mut self, f: F) -> &mut T
    where
        F: FnOnce(&mut T),
    {
        self.inner.or_update(f)
    }

    fn degrade(&mut self) {
        self.inner.degrade()
    }

    fn state(&self) -> TriState {
        self.inner.state()
    }
}

impl<T> std::fmt::Debug for Quality<T>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt(f)
    }
}

pub enum QualityEnum<T> {
    Transitional,
    Bad(T),
    Good(T),
}

impl<T> QualityEnum<T> {
    const fn good(&self) -> Option<&T> {
        match self {
            Self::Transitional => unreachable!(),
            Self::Bad(_) => None,
            Self::Good(value) => Some(value),
        }
    }

    const fn good_mut(&mut self) -> Option<&mut T> {
        match self {
            Self::Transitional => unreachable!(),
            Self::Bad(_) => None,
            Self::Good(value) => Some(value),
        }
    }

    const fn bad(&self) -> Option<&T> {
        match self {
            Self::Transitional => unreachable!(),
            Self::Bad(value) => Some(value),
            Self::Good(_) => None,
        }
    }

    const fn bad_mut(&mut self) -> Option<&mut T> {
        match self {
            Self::Transitional => unreachable!(),
            Self::Bad(value) => Some(value),
            Self::Good(_) => None,
        }
    }

    fn or_update<F>(&mut self, f: F) -> &mut T
    where
        F: FnOnce(&mut T),
    {
        *self = match std::mem::replace(self, Self::Transitional) {
            Self::Transitional => unreachable!(),
            Self::Bad(mut value) => {
                f(&mut value);
                Self::Good(value)
            }
            Self::Good(value) => Self::Good(value),
        };
        self.good_mut().expect("set above")
    }

    fn degrade(&mut self) {
        *self = match std::mem::replace(self, Self::Transitional) {
            Self::Transitional => unreachable!(),
            Self::Good(value) => Self::Bad(value),
            Self::Bad(value) => Self::Bad(value),
        }
    }

    fn state(&self) -> TriState {
        match self {
            Self::Transitional => unreachable!(),
            Self::Bad(_) => TriState::Bad,
            Self::Good(_) => TriState::Good,
        }
    }
}

impl<T> std::fmt::Debug for QualityEnum<T>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transitional => write!(f, "Transitional"),
            Self::Bad(arg0) => f.debug_tuple("Bad").field(arg0).finish(),
            Self::Good(arg0) => f.debug_tuple("Good").field(arg0).finish(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum TriState {
    Missing,
    Good,
    Bad,
}
